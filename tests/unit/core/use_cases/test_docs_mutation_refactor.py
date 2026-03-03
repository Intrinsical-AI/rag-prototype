from __future__ import annotations

import hashlib
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from local_rag_backend.core.domain.profiles import StorageProfileRegistry
from local_rag_backend.core.ports.contracts import DocsMutationPorts, MutationRecord
from local_rag_backend.core.use_cases.docs_mutation import (
    MutationCoordinator,
    MutationSagaExecutor,
    build_journal,
)
from local_rag_backend.core.use_cases.docs_mutation_contracts import (
    MutationIntent,
    MutationUpsertInput,
    intent_to_dict,
    normalize_intent,
    summary_from_record,
    summary_to_payload,
)
from local_rag_backend.core.use_cases.results import MutationSummary, UpsertDocResult


@dataclass
class _SettingsStub:
    retrieval_mode: str = "sparse"
    vector_backend: str = "faiss"
    storage_profile: str = ""
    write_lock_timeout_s: float = 1.0
    write_lock_poll_s: float = 0.01
    mutation_batch_max_size: int = 32
    mutation_batch_max_wait_ms: int = 50
    index_path: str = "unused.idx"
    id_map_path: str = "unused.idmap"

    def get_coordination_dir(self) -> Path:
        return Path(".")


class _MemoryJournal:
    def __init__(self) -> None:
        self._records: dict[str, MutationRecord] = {}
        self._lock = threading.Lock()

    def get(self, op_id: str) -> MutationRecord | None:
        with self._lock:
            return self._records.get(op_id)

    def upsert(self, record: MutationRecord) -> None:
        with self._lock:
            self._records[record.op_id] = record

    def delete(self, op_id: str) -> None:
        with self._lock:
            self._records.pop(op_id, None)

    def list_incomplete(self, *, limit: int = 100) -> list[MutationRecord]:
        states = {"PREPARED", "SQL_COMMITTED", "COMPENSATING", "FAILED_NEEDS_RECOVERY"}
        with self._lock:
            return [r for r in self._records.values() if r.state in states][:limit]


class _UpsertResult:
    def __init__(
        self,
        *,
        external_id: str,
        id_: str,
        action: str = "inserted",
        content_changed: bool = True,
    ) -> None:
        self.external_id = external_id
        self.id = id_
        self.action = action
        self.content_changed = content_changed


class _DocRepoSparseStub:
    def get_tombstoned_external_ids(self, external_ids: list[str]) -> set[str]:
        return set()

    def upsert_documents_by_external_id(
        self, items: list[object]
    ) -> tuple[list[_UpsertResult], list[tuple[str, str]], list[str]]:
        result = _UpsertResult(
            external_id="doc-1", id_="1", action="inserted", content_changed=True
        )
        return [result], [], []

    def get(self, ids: list[str]) -> list[Any]:
        return []

    def delete_documents(self, ids: list[str]) -> None:
        return None

    def delete_by_external_ids(
        self, external_ids: list[str]
    ) -> tuple[int, list[str], list[str], int]:
        return 0, [], [], 0


class _DocRepoDenseFailingVecStub(_DocRepoSparseStub):
    def upsert_documents_by_external_id(
        self, items: list[object]
    ) -> tuple[list[_UpsertResult], list[tuple[str, str]], list[str]]:
        first = items[0] if items else {}
        if isinstance(first, dict):
            ext = str(first.get("external_id", "doc-x"))
            content = str(first.get("content", "hello dense"))
        else:
            ext = str(getattr(first, "external_id", "doc-x"))
            content = str(getattr(first, "content", "hello dense"))
        result = _UpsertResult(external_id=ext, id_="42", action="inserted", content_changed=True)
        return [result], [("42", content)], []

    def hard_delete_by_external_ids(self, external_ids: list[str]) -> None:
        return None

    def restore_from_snapshots(self, snapshots: list[dict[str, Any]]) -> None:
        return None

    def snapshot_by_external_ids(self, external_ids: list[str]) -> list[dict[str, Any]]:
        _ = external_ids
        return []


class _DocRepoDenseRollbackFailStub(_DocRepoDenseFailingVecStub):
    def hard_delete_by_external_ids(self, external_ids: list[str]) -> None:
        raise RuntimeError("rollback fails")


class _FailingVectorRepo:
    def apply_delta_atomic(
        self, *, delete_ids: list[str], upserts: list[tuple[str, list[float]]]
    ) -> None:
        raise RuntimeError("vector fails")


class _DenseEmbedder:
    dim = 1

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


@dataclass(frozen=True)
class _DomainDoc:
    id: str
    content: str
    external_id: str | None = None


class _RecoveringDocRepo:
    def __init__(self) -> None:
        self._docs_by_external_id: dict[str, _DomainDoc] = {
            "doc-recover": _DomainDoc(id="42", content="payload", external_id="doc-recover")
        }
        self._docs_by_id: dict[str, _DomainDoc] = {"42": self._docs_by_external_id["doc-recover"]}

    def get_tombstoned_external_ids(self, external_ids: list[str]) -> set[str]:
        return set()

    def upsert_documents_by_external_id(
        self, items: list[object]
    ) -> tuple[list[_UpsertResult], list[tuple[str, str]], list[str]]:
        raise AssertionError("not needed for this recovery test")

    def get(self, ids: list[str]) -> list[Any]:
        out: list[Any] = []
        for doc_id in ids:
            doc = self._docs_by_id.get(str(doc_id))
            if doc is not None:
                out.append(doc)
        return out

    def delete_documents(self, ids: list[str]) -> None:
        for doc_id in ids:
            row = self._docs_by_id.pop(str(doc_id), None)
            if row is not None and row.external_id is not None:
                self._docs_by_external_id.pop(str(row.external_id), None)

    def delete_by_external_ids(
        self, external_ids: list[str]
    ) -> tuple[int, list[str], list[str], int]:
        raise AssertionError("not needed for this recovery test")

    def hard_delete_by_external_ids(self, external_ids: list[str]) -> None:
        for ext in external_ids:
            row = self._docs_by_external_id.pop(str(ext), None)
            if row is not None:
                self._docs_by_id.pop(str(row.id), None)

    def restore_from_snapshots(self, snapshots: list[dict[str, Any]]) -> None:
        for snap in snapshots:
            doc_id = str(snap.get("id") or "").strip()
            ext_id_raw = snap.get("external_id")
            if not doc_id:
                continue
            row = _DomainDoc(
                id=doc_id,
                content=str(snap.get("content") or ""),
                external_id=(str(ext_id_raw) if ext_id_raw is not None else None),
            )
            self._docs_by_id[doc_id] = row
            if row.external_id is not None:
                self._docs_by_external_id[row.external_id] = row

    def get_all_documents(self) -> list[_DomainDoc]:
        return list(self._docs_by_id.values())


class _VectorRepoForRecovery:
    def __init__(self) -> None:
        self.ids = {"42"}

    def apply_delta_atomic(
        self, *, delete_ids: list[str], upserts: list[tuple[str, list[float]]]
    ) -> None:
        return None

    def rebuild(self, ids: list[str], vectors: list[list[float]]) -> None:
        _ = vectors
        self.ids = {str(x) for x in ids}


def _noop_write_lock(*, coordination_dir=None, timeout_s=None, poll_s=None):  # type: ignore[no-untyped-def]
    return nullcontext()


def _build_ports(
    *,
    doc_repo: Any,
    uow_factory: Any | None = None,
    retrieval_mode: str = "sparse",
    vector_repo_factory: Any | None = None,
) -> tuple[MutationCoordinator, _MemoryJournal]:
    journal = _MemoryJournal()
    settings = _SettingsStub(retrieval_mode=retrieval_mode)
    ports = DocsMutationPorts(
        build_embedder=lambda: _DenseEmbedder(),
        doc_repo_factory=lambda: doc_repo,
        build_upsert_doc=lambda **kwargs: dict(kwargs),
        vector_repo_factory=(vector_repo_factory or (lambda **kwargs: object())),
        rebuild_fn=lambda **kwargs: 0,
        write_lock=_noop_write_lock,
        mutation_journal_factory=lambda: journal,
        storage_profile_registry=StorageProfileRegistry(),
        mutation_uow_factory=uow_factory,
    )
    return MutationCoordinator(settings_obj=settings, ports=ports), journal


def test_contract_helpers_normalize_and_payload_roundtrip() -> None:
    normalized = normalize_intent(
        intent=MutationIntent(
            op_id="  ",
            upserts=(
                MutationUpsertInput(external_id=" doc-1 ", content=" hello "),
                MutationUpsertInput(external_id="doc-2", content=""),
            ),
            delete_ids=(" id-1 ", "id-1"),
            delete_external_ids=(" ext-1 ", "ext-1"),
            source=" api ",
        ),
        new_op_id=lambda: "mut:new",
    )
    assert normalized.op_id == "mut:new"
    assert normalized.source == " api "
    assert [u.external_id for u in normalized.upserts] == ["doc-1"]
    assert list(normalized.delete_ids) == ["id-1"]
    assert list(normalized.delete_external_ids) == ["ext-1"]

    summary = MutationSummary(
        op_id="mut:1",
        inserted=1,
        results=[
            UpsertDocResult(external_id="doc-1", id="1", action="inserted", content_changed=True)
        ],
    )
    record = MutationRecord(
        op_id="mut:1",
        state="COMMITTED",
        intent=intent_to_dict(normalized),
        outcome=summary_to_payload(summary),
    )
    hydrated = summary_from_record(record)
    assert hydrated.op_id == "mut:1"
    assert hydrated.inserted == 1
    assert hydrated.results is not None and hydrated.results[0].external_id == "doc-1"


def test_mutation_uow_wraps_sql_mutation_path() -> None:
    events: list[str] = []

    @contextmanager
    def _uow():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    coordinator, _journal = _build_ports(doc_repo=_DocRepoSparseStub(), uow_factory=_uow)
    summary = coordinator.execute(
        MutationIntent(
            op_id="mut:uow-ok", upserts=(MutationUpsertInput(external_id="doc-1", content="hello"),)
        )
    )

    assert summary.inserted == 1
    assert events == ["enter", "exit"]


def test_mutation_uow_wraps_sql_and_rollback_on_vector_failure() -> None:
    events: list[str] = []

    @contextmanager
    def _uow():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    coordinator, _journal = _build_ports(
        doc_repo=_DocRepoDenseFailingVecStub(),
        uow_factory=_uow,
        retrieval_mode="dense",
        vector_repo_factory=lambda **kwargs: _FailingVectorRepo(),
    )
    with pytest.raises(RuntimeError, match="vector fails"):
        coordinator.execute(
            MutationIntent(
                op_id="mut:uow-rollback",
                upserts=(MutationUpsertInput(external_id="doc-x", content="hello dense"),),
            )
        )

    # One UoW for SQL mutation and one for rollback compensation.
    assert events == ["enter", "exit", "enter", "exit"]


def test_vector_and_rollback_failure_mark_failed_needs_recovery() -> None:
    coordinator, journal = _build_ports(
        doc_repo=_DocRepoDenseRollbackFailStub(),
        retrieval_mode="dense",
        vector_repo_factory=lambda **kwargs: _FailingVectorRepo(),
    )

    with pytest.raises(
        RuntimeError,
        match=r"Mutation failed after SQL commit and rollback did not complete\.",
    ):
        coordinator.execute(
            MutationIntent(
                op_id="mut:failed-recovery",
                upserts=(MutationUpsertInput(external_id="doc-rb", content="boom"),),
            )
        )

    record = journal.get("mut:failed-recovery")
    assert record is not None
    assert record.state == "FAILED_NEEDS_RECOVERY"
    assert record.error is not None
    assert "vector=vector fails" in record.error
    assert "rollback=rollback fails" in record.error


def test_recover_incomplete_keeps_failed_state_when_rollback_still_fails() -> None:
    coordinator, journal = _build_ports(doc_repo=_DocRepoDenseRollbackFailStub())

    intent = MutationIntent(
        op_id="mut:recover-fails",
        upserts=(MutationUpsertInput(external_id="doc-rf", content="hello"),),
    )
    journal.upsert(
        MutationRecord(
            op_id=intent.op_id,
            state="SQL_COMMITTED",
            intent=intent_to_dict(intent),
            before_image={"docs": [], "existing_tombstones": []},
        )
    )

    repaired = coordinator.recover_incomplete(limit=10)
    assert repaired == 1
    recovered = journal.get(intent.op_id)
    assert recovered is not None
    assert recovered.state == "FAILED_NEEDS_RECOVERY"
    assert recovered.error == "rollback fails"


def test_recover_incomplete_without_before_image_marks_rolled_back() -> None:
    coordinator, journal = _build_ports(doc_repo=_DocRepoSparseStub())

    intent = MutationIntent(
        op_id="mut:no-before-image",
        upserts=(MutationUpsertInput(external_id="doc-nbi", content="hello"),),
    )
    journal.upsert(
        MutationRecord(
            op_id=intent.op_id,
            state="SQL_COMMITTED",
            intent=intent_to_dict(intent),
            before_image=None,
        )
    )

    repaired = coordinator.recover_incomplete(limit=10)
    assert repaired == 1
    recovered = journal.get(intent.op_id)
    assert recovered is not None
    assert recovered.state == "ROLLED_BACK"
    assert recovered.error is not None
    assert "No before_image was available" in recovered.error


def test_recover_incomplete_dense_reconciles_vector_index_with_sql_state() -> None:
    journal = _MemoryJournal()
    doc_repo = _RecoveringDocRepo()
    vec_repo = _VectorRepoForRecovery()
    settings = _SettingsStub(retrieval_mode="dense")

    def _rebuild_from_sql(*, doc_repo, vec_repo, embedder):
        _ = embedder
        docs = list(doc_repo.get_all_documents())
        ids = [d.id for d in docs]
        vectors = [[0.0] for _ in ids]
        vec_repo.rebuild(ids, vectors)
        return len(ids)

    ports = DocsMutationPorts(
        build_embedder=lambda: _DenseEmbedder(),
        doc_repo_factory=lambda: doc_repo,
        build_upsert_doc=lambda **kwargs: dict(kwargs),
        vector_repo_factory=lambda **kwargs: vec_repo,
        rebuild_fn=_rebuild_from_sql,
        write_lock=_noop_write_lock,
        mutation_journal_factory=lambda: journal,
        storage_profile_registry=StorageProfileRegistry(),
        mutation_uow_factory=None,
    )
    coordinator = MutationCoordinator(settings_obj=settings, ports=ports)

    intent = MutationIntent(
        op_id="mut:recover-dense",
        upserts=(MutationUpsertInput(external_id="doc-recover", content="payload"),),
    )
    journal.upsert(
        MutationRecord(
            op_id=intent.op_id,
            state="SQL_COMMITTED",
            intent=intent_to_dict(intent),
            before_image={"docs": [], "existing_tombstones": []},
        )
    )

    assert vec_repo.ids == {"42"}
    repaired = coordinator.recover_incomplete(limit=10)
    assert repaired == 1

    recovered = journal.get(intent.op_id)
    assert recovered is not None
    assert recovered.state == "ROLLED_BACK"
    assert doc_repo.get_all_documents() == []
    assert vec_repo.ids == set()


def test_dense_embeddings_are_precomputed_outside_write_lock() -> None:
    lock_state = {"held": False}

    @contextmanager
    def _write_lock(*, coordination_dir=None, timeout_s=None, poll_s=None):
        _ = coordination_dir, timeout_s, poll_s
        lock_state["held"] = True
        try:
            yield
        finally:
            lock_state["held"] = False

    class _Embedder:
        dim = 1

        def __init__(self) -> None:
            self.calls = 0

        def embed(self, texts: list[str]) -> list[list[float]]:
            if lock_state["held"]:
                raise AssertionError("embed() must not run while write lock is held")
            self.calls += 1
            return [[1.0] for _ in texts]

    class _DenseDocRepo(_DocRepoDenseFailingVecStub):
        def get_existing_doc_states_by_external_id(self, external_ids: list[str]) -> dict[str, Any]:
            _ = external_ids
            return {}

    class _Vec:
        def apply_delta_atomic(self, *, delete_ids, upserts):
            _ = delete_ids, upserts
            return None

    embedder = _Embedder()
    settings = _SettingsStub(retrieval_mode="dense")
    journal = _MemoryJournal()
    ports = DocsMutationPorts(
        build_embedder=lambda: embedder,
        doc_repo_factory=lambda: _DenseDocRepo(),
        build_upsert_doc=lambda **kwargs: dict(kwargs),
        vector_repo_factory=lambda **kwargs: _Vec(),
        rebuild_fn=lambda **kwargs: 0,
        write_lock=_write_lock,
        mutation_journal_factory=lambda: journal,
        storage_profile_registry=StorageProfileRegistry(),
        mutation_uow_factory=None,
    )
    coordinator = MutationCoordinator(settings_obj=settings, ports=ports)
    out = coordinator.execute(
        MutationIntent(
            op_id="mut:precompute-outside-lock",
            upserts=(MutationUpsertInput(external_id="doc-1", content="hello"),),
        )
    )
    assert out.inserted == 1
    assert embedder.calls == 1


def test_dense_mutation_requires_rollback_capable_repo_contract() -> None:
    class _MissingRollbackRepo(_DocRepoSparseStub):
        pass

    coordinator, _journal = _build_ports(doc_repo=_MissingRollbackRepo(), retrieval_mode="dense")
    with pytest.raises(RuntimeError, match="rollback contract"):
        coordinator.execute(
            MutationIntent(
                op_id="mut:missing-rollback-contract",
                upserts=(MutationUpsertInput(external_id="doc-1", content="hello"),),
            )
        )


def test_stale_precompute_is_safe_when_document_changes_before_lock() -> None:
    @dataclass(frozen=True)
    class _ExistingState:
        content: str
        content_sha256: str | None

    class _Repo:
        def __init__(self) -> None:
            self._content_by_external_id: dict[str, str] = {"doc-1": "v0"}
            self._id_by_external_id: dict[str, str] = {"doc-1": "1"}

        def get_tombstoned_external_ids(self, external_ids: list[str]) -> set[str]:
            _ = external_ids
            return set()

        def get_existing_doc_states_by_external_id(self, external_ids: list[str]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for external_id in external_ids:
                content = self._content_by_external_id.get(external_id)
                if content is None:
                    continue
                out[external_id] = _ExistingState(
                    content=content,
                    content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                )
            return out

        def snapshot_by_external_ids(self, external_ids: list[str]) -> list[dict[str, Any]]:
            snapshots: list[dict[str, Any]] = []
            for external_id in external_ids:
                content = self._content_by_external_id.get(external_id)
                if content is None:
                    continue
                snapshots.append(
                    {
                        "id": self._id_by_external_id[external_id],
                        "external_id": external_id,
                        "content": content,
                    }
                )
            return snapshots

        def hard_delete_by_external_ids(self, external_ids: list[str]) -> None:
            for external_id in external_ids:
                self._content_by_external_id.pop(external_id, None)

        def restore_from_snapshots(self, snapshots: list[dict[str, Any]]) -> None:
            for snap in snapshots:
                external_id = str(snap.get("external_id") or "").strip()
                content = str(snap.get("content") or "")
                if not external_id:
                    continue
                self._content_by_external_id[external_id] = content

        def upsert_documents_by_external_id(
            self, items: list[object]
        ) -> tuple[list[_UpsertResult], list[tuple[str, str]], list[str]]:
            out: list[_UpsertResult] = []
            changed: list[tuple[str, str]] = []
            updated_ids: list[str] = []
            for item in items:
                payload = dict(item) if isinstance(item, dict) else item.__dict__
                external_id = str(payload["external_id"])
                content = str(payload["content"])
                existing = self._content_by_external_id.get(external_id)
                if existing is None:
                    doc_id = str(len(self._id_by_external_id) + 1)
                    self._id_by_external_id[external_id] = doc_id
                    self._content_by_external_id[external_id] = content
                    out.append(
                        _UpsertResult(
                            external_id=external_id,
                            id_=doc_id,
                            action="inserted",
                            content_changed=True,
                        )
                    )
                    changed.append((doc_id, content))
                    continue
                if existing == content:
                    out.append(
                        _UpsertResult(
                            external_id=external_id,
                            id_=self._id_by_external_id[external_id],
                            action="unchanged",
                            content_changed=False,
                        )
                    )
                    continue
                self._content_by_external_id[external_id] = content
                doc_id = self._id_by_external_id[external_id]
                out.append(
                    _UpsertResult(
                        external_id=external_id,
                        id_=doc_id,
                        action="updated",
                        content_changed=True,
                    )
                )
                changed.append((doc_id, content))
                updated_ids.append(doc_id)
            return out, changed, updated_ids

        def get(self, ids: list[str]) -> list[Any]:
            _ = ids
            return []

        def delete_documents(self, ids: list[str]) -> None:
            _ = ids

        def delete_by_external_ids(
            self, external_ids: list[str]
        ) -> tuple[int, list[str], list[str], int]:
            _ = external_ids
            return 0, [], [], 0

    class _VecRepo:
        def apply_delta_atomic(self, *, delete_ids, upserts):
            _ = delete_ids, upserts
            return None

    repo = _Repo()
    settings = _SettingsStub(retrieval_mode="dense")
    journal = _MemoryJournal()
    ports = DocsMutationPorts(
        build_embedder=lambda: _DenseEmbedder(),
        doc_repo_factory=lambda: repo,
        build_upsert_doc=lambda **kwargs: dict(kwargs),
        vector_repo_factory=lambda **kwargs: _VecRepo(),
        rebuild_fn=lambda **kwargs: 0,
        write_lock=_noop_write_lock,
        mutation_journal_factory=lambda: journal,
        storage_profile_registry=StorageProfileRegistry(),
        mutation_uow_factory=None,
    )
    saga = MutationSagaExecutor(settings_obj=settings, ports=ports)
    file_journal = build_journal(ports=ports)

    stale_intent = MutationIntent(
        op_id="mut:stale-precompute",
        upserts=(MutationUpsertInput(external_id="doc-1", content="v0"),),
    )
    stale_precomputed = saga.precompute_vectors_for_intent(
        intent=stale_intent,
        vector_mode_enabled=True,
    )

    first_intent = MutationIntent(
        op_id="mut:first-update",
        upserts=(MutationUpsertInput(external_id="doc-1", content="v1"),),
    )
    first_precomputed = saga.precompute_vectors_for_intent(
        intent=first_intent,
        vector_mode_enabled=True,
    )

    class _Prepared:
        def __init__(self, *, intent, vectors):
            self.intent = intent
            self.vector_mode_enabled = True
            self.precomputed_vectors_by_external_id = vectors

    saga.execute_locked(
        prepared=_Prepared(intent=first_intent, vectors=first_precomputed),
        journal=file_journal,
    )
    saga.execute_locked(
        prepared=_Prepared(intent=stale_intent, vectors=stale_precomputed),
        journal=file_journal,
    )

    assert repo._content_by_external_id["doc-1"] == "v0"


def test_batch_drain_acquires_lock_once_for_two_concurrent_mutations() -> None:
    class _Repo:
        def __init__(self) -> None:
            self._next = 1
            self._lock = threading.Lock()

        def get_tombstoned_external_ids(self, external_ids: list[str]) -> set[str]:
            _ = external_ids
            return set()

        def upsert_documents_by_external_id(self, items: list[object]):
            out = []
            changed: list[tuple[str, str]] = []
            with self._lock:
                for item in items:
                    if isinstance(item, dict):
                        external_id = str(item.get("external_id"))
                        content = str(item.get("content"))
                    else:
                        external_id = str(item.external_id)
                        content = str(item.content)
                    doc_id = str(self._next)
                    self._next += 1
                    out.append(
                        _UpsertResult(
                            external_id=external_id,
                            id_=doc_id,
                            action="inserted",
                            content_changed=True,
                        )
                    )
                    changed.append((doc_id, content))
            return out, changed, []

        def get(self, ids: list[str]) -> list[Any]:
            _ = ids
            return []

        def delete_documents(self, ids: list[str]) -> None:
            _ = ids
            return None

        def delete_by_external_ids(
            self, external_ids: list[str]
        ) -> tuple[int, list[str], list[str], int]:
            _ = external_ids
            return 0, [], [], 0

    lock_entries = 0
    lock_guard = threading.Lock()

    @contextmanager
    def _write_lock(*, coordination_dir=None, timeout_s=None, poll_s=None):
        _ = coordination_dir, timeout_s, poll_s
        nonlocal lock_entries
        with lock_guard:
            lock_entries += 1
        yield

    settings = _SettingsStub(
        retrieval_mode="sparse",
        mutation_batch_max_size=32,
        mutation_batch_max_wait_ms=200,
    )
    repo = _Repo()
    journal = _MemoryJournal()
    ports = DocsMutationPorts(
        build_embedder=lambda: _DenseEmbedder(),
        doc_repo_factory=lambda: repo,
        build_upsert_doc=lambda **kwargs: dict(kwargs),
        vector_repo_factory=lambda **kwargs: object(),
        rebuild_fn=lambda **kwargs: 0,
        write_lock=_write_lock,
        mutation_journal_factory=lambda: journal,
        storage_profile_registry=StorageProfileRegistry(),
        mutation_uow_factory=None,
    )
    coordinator = MutationCoordinator(settings_obj=settings, ports=ports)

    start = threading.Barrier(3)
    results: list[MutationSummary] = []
    errors: list[Exception] = []

    def _worker(op_id: str, external_id: str) -> None:
        start.wait()
        try:
            summary = coordinator.execute(
                MutationIntent(
                    op_id=op_id,
                    upserts=(MutationUpsertInput(external_id=external_id, content="payload"),),
                )
            )
            results.append(summary)
        except Exception as exc:  # pragma: no cover - assertion below fails if this triggers
            errors.append(exc)

    t1 = threading.Thread(target=_worker, args=("mut:b1", "doc-b1"), daemon=True)
    t2 = threading.Thread(target=_worker, args=("mut:b2", "doc-b2"), daemon=True)
    t1.start()
    t2.start()
    start.wait()
    t1.join(timeout=3)
    t2.join(timeout=3)

    assert not errors
    assert len(results) == 2
    assert all(r.inserted == 1 for r in results)
    assert lock_entries == 1

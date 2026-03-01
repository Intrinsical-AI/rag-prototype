from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from local_rag_backend.core.domain.profiles import StorageProfileRegistry
from local_rag_backend.core.ports.contracts import DocsMutationPorts, MutationRecord
from local_rag_backend.core.use_cases.docs_mutation import MutationCoordinator
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
    index_path: str = "unused.idx"
    id_map_path: str = "unused.idmap"

    def get_coordination_dir(self) -> Path:
        return Path(".")


class _MemoryJournal:
    def __init__(self) -> None:
        self._records: dict[str, MutationRecord] = {}

    def get(self, op_id: str) -> MutationRecord | None:
        return self._records.get(op_id)

    def upsert(self, record: MutationRecord) -> None:
        self._records[record.op_id] = record

    def delete(self, op_id: str) -> None:
        self._records.pop(op_id, None)

    def list_incomplete(self, *, limit: int = 100) -> list[MutationRecord]:
        states = {"PREPARED", "SQL_COMMITTED", "COMPENSATING", "FAILED_NEEDS_RECOVERY"}
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
        result = _UpsertResult(
            external_id="doc-x", id_="42", action="inserted", content_changed=True
        )
        return [result], [("42", "hello dense")], []

    def hard_delete_by_external_ids(self, external_ids: list[str]) -> None:
        return None

    def restore_from_snapshots(self, snapshots: list[dict[str, Any]]) -> None:
        return None


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

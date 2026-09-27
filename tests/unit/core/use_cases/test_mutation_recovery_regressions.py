from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from threading import Barrier, Event, RLock
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from local_rag_backend.core.domain.embeddings import EmbeddingIdentity
from local_rag_backend.core.domain.profiles import StorageProfileRegistry
from local_rag_backend.core.errors import MutationRecoveryRequiredError
from local_rag_backend.core.ports.contracts import (
    DocsMutationPorts,
    IndexMutationPorts,
    MutationRecord,
)
from local_rag_backend.core.services.maintenance import rebuild_index_from_db
from local_rag_backend.core.use_cases._batch_coordinator import MutationBatchItem
from local_rag_backend.core.use_cases._mutation_saga_executor import PreparedMutation
from local_rag_backend.core.use_cases.docs_mutation import MutationCoordinator
from local_rag_backend.core.use_cases.docs_mutation_contracts import (
    MutationIntent,
    MutationUpsertInput,
    intent_to_dict,
)
from local_rag_backend.core.use_cases.index import rebuild_index_sync
from local_rag_backend.infrastructure.persistence.shared.mutation_journal import FileMutationJournal
from local_rag_backend.infrastructure.persistence.sql.base import Base, session_uow
from local_rag_backend.infrastructure.persistence.sql.document_storage import SqlDocumentStorage
from local_rag_backend.infrastructure.persistence.vector import index as index_module
from local_rag_backend.infrastructure.persistence.vector.manifest import purge_index_artifacts
from local_rag_backend.infrastructure.persistence.vector.storage import VectorStorage
from local_rag_backend.settings import Settings


class _Embedder:
    dim = 2
    identity = EmbeddingIdentity("sentence_transformers", "test", 2, synthetic=True)

    def embed(self, texts):
        return [[1.0, 0.0] if text == "v1" else [0.0, 1.0] for text in texts]


@pytest.fixture
def runtime(tmp_path):
    engine = create_engine("sqlite:///" + str(tmp_path / "db.sqlite"))
    sessions = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    repo = SqlDocumentStorage(sessions)
    cfg = Settings(
        data_dir=tmp_path,
        retrieval_mode="dense",
        vector_backend="numpy",
        index_path=str(tmp_path / "index.npy"),
        id_map_path=str(tmp_path / "ids.json"),
        mutation_batch_max_wait_ms=0,
        st_embedding_model="test",
        synthetic_embeddings=True,
    )
    journal = FileMutationJournal(tmp_path / "journal")
    lock = RLock()

    @contextmanager
    def write_lock(**kwargs):
        with lock:
            yield

    def vectors(**kwargs):
        kwargs.setdefault("settings_obj", cfg)
        kwargs.setdefault("embedding_identity", _Embedder.identity)
        return VectorStorage(**kwargs)

    rebuilds = []
    index_ports = IndexMutationPorts(
        build_embedder=_Embedder,
        doc_repo_factory=lambda: repo,
        vector_repo_factory=vectors,
        purge_index_artifacts_fn=purge_index_artifacts,
        rebuild_fn=rebuild_index_from_db,
    )

    def reconcile():
        rebuilds.append(True)
        return rebuild_index_sync(settings_obj=cfg, ports=index_ports)

    ports = DocsMutationPorts(
        build_embedder=_Embedder,
        doc_repo_factory=lambda: repo,
        build_upsert_doc=repo.UpsertDoc,
        vector_repo_factory=vectors,
        reconcile_index=reconcile,
        write_lock=write_lock,
        mutation_journal_factory=lambda: journal,
        storage_profile_registry=StorageProfileRegistry(),
        mutation_uow_factory=lambda: session_uow(session_factory=sessions),
    )
    result = SimpleNamespace(
        repo=repo,
        settings=cfg,
        ports=ports,
        journal=journal,
        rebuilds=rebuilds,
        coordinator=MutationCoordinator(settings_obj=cfg, ports=ports),
        vector=lambda: vectors(
            index_path=cfg.index_path, id_map_path=cfg.id_map_path, dim=2, backend="numpy"
        ),
    )
    yield result
    engine.dispose()


def _upsert(op_id, content="v1", external_id="doc"):
    return MutationIntent(
        op_id=op_id, upserts=(MutationUpsertInput(external_id=external_id, content=content),)
    )


def test_committed_replay_preserves_later_content_and_original_response(runtime):
    a = _upsert("a")
    original = runtime.coordinator.execute(a)
    runtime.coordinator.execute(_upsert("b", "v2"))
    assert runtime.coordinator.execute(a) == original
    assert runtime.repo.get_all_documents()[0].content == "v2"
    assert runtime.rebuilds == []


def test_same_op_id_concurrent_requests_return_one_original_outcome(runtime):
    barrier = Barrier(2)
    original_embed = runtime.ports.build_embedder

    class Embedder(_Embedder):
        def embed(self, texts):
            barrier.wait(timeout=5)
            return original_embed().embed(texts)

    coordinator = MutationCoordinator(
        settings_obj=runtime.settings, ports=replace(runtime.ports, build_embedder=Embedder)
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(lambda _: coordinator.execute(_upsert("same")), range(2)))
    assert a == b
    assert a.inserted == 1
    assert len(runtime.repo.get_all_documents()) == 1
    assert runtime.vector().ntotal == 1


@pytest.mark.parametrize("mismatch", [False, True])
def test_fast_replay_checks_intent_and_outcome_under_lock_without_embedding(runtime, mismatch):
    original = runtime.coordinator.execute(_upsert("replay"))
    locks = []

    @contextmanager
    def observe_lock(**kwargs):
        with runtime.ports.write_lock(**kwargs):
            locks.append(True)
            yield

    def unexpected_embedding():
        raise AssertionError("replay must not construct an embedder")

    coordinator = MutationCoordinator(
        settings_obj=runtime.settings,
        ports=replace(runtime.ports, write_lock=observe_lock, build_embedder=unexpected_embedding),
    )
    if mismatch:
        with pytest.raises(ValueError, match="replay mismatch"):
            coordinator.execute(_upsert("replay", "v2"))
    else:
        assert coordinator.execute(_upsert("replay")) == original
    assert locks == [True]


def test_fast_replay_reloads_commit_completed_by_other_thread_while_waiting_for_lock(runtime):
    intent = _upsert("writer")
    original = runtime.coordinator.execute(intent)
    committed = runtime.journal.get(intent.op_id)
    runtime.journal.upsert(replace(committed, state="SQL_COMMITTED"))
    waiting_for_lock = Event()

    @contextmanager
    def observe_lock(**kwargs):
        waiting_for_lock.set()
        with runtime.ports.write_lock(**kwargs):
            yield

    def unexpected_embedding():
        raise AssertionError("concurrent commit must be replayed without embedding")

    coordinator = MutationCoordinator(
        settings_obj=runtime.settings,
        ports=replace(runtime.ports, write_lock=observe_lock, build_embedder=unexpected_embedding),
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        with runtime.ports.write_lock():
            future = pool.submit(coordinator.execute, intent)
            assert waiting_for_lock.wait(timeout=5)
            runtime.journal.upsert(committed)
        assert future.result(timeout=5) == original
    assert runtime.repo.get_all_documents()[0].content == "v1"
    assert runtime.rebuilds == []


def test_recovery_enumerates_only_after_acquiring_lock(runtime):
    runtime.coordinator.execute(_upsert("initial"))
    before = {"docs": runtime.repo.snapshot_by_external_ids(["doc"]), "existing_tombstones": []}
    runtime.coordinator.execute(_upsert("writer", "v2"))
    committed = runtime.journal.get("writer")
    runtime.journal.upsert(replace(committed, state="SQL_COMMITTED", before_image=before))

    @contextmanager
    def finish_writer_before_lock(**kwargs):
        runtime.journal.upsert(committed)
        yield

    coordinator = MutationCoordinator(
        settings_obj=runtime.settings,
        ports=replace(runtime.ports, write_lock=finish_writer_before_lock),
    )
    assert coordinator.recover_incomplete() == 0
    assert runtime.repo.get_all_documents()[0].content == "v2"
    assert runtime.journal.get("writer").state == "COMMITTED"


@pytest.mark.parametrize("operation", ["update", "insert", "delete"])
def test_partial_vector_file_write_recovers_both_stores(runtime, monkeypatch, operation):
    runtime.coordinator.execute(_upsert("initial"))
    original = index_module.save_id_map_json
    failed = False

    def fail_once(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("id-map write failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(index_module, "save_id_map_json", fail_once)
    intent = {
        "update": _upsert("failed", "v2"),
        "insert": _upsert("failed", "v2", "extra"),
        "delete": MutationIntent(op_id="failed", delete_external_ids=("doc",)),
    }[operation]
    with pytest.raises(OSError, match="id-map write failed"):
        runtime.coordinator.execute(intent)
    assert [(d.external_id, d.content) for d in runtime.repo.get_all_documents()] == [("doc", "v1")]
    assert runtime.vector().ntotal == 1
    assert runtime.vector().search([1.0, 0.0], 1)[1].tolist() == [0.0]
    assert runtime.journal.get("failed").state == "ROLLED_BACK"
    assert runtime.coordinator.recover_incomplete() == 0
    assert len(runtime.rebuilds) == 1


def test_pre_vector_failure_does_not_rebuild(runtime):
    runtime.coordinator.execute(_upsert("initial"))

    def fail_before_vector(**kwargs):
        raise RuntimeError("cannot construct vector adapter")

    coordinator = MutationCoordinator(
        settings_obj=runtime.settings,
        ports=replace(runtime.ports, vector_repo_factory=fail_before_vector),
    )
    with pytest.raises(RuntimeError, match="cannot construct"):
        coordinator.execute(_upsert("failed", "v2"))
    assert runtime.rebuilds == []
    assert runtime.repo.get_all_documents()[0].content == "v1"
    assert runtime.journal.get("failed").vector_attempted is False
    assert runtime.journal.get("failed").state == "ROLLED_BACK"


def test_failed_rebuild_blocks_rest_of_batch_without_repeated_rebuilds(runtime, monkeypatch):
    runtime.coordinator.execute(_upsert("initial"))
    attempts = []

    def fail_rebuild():
        attempts.append(True)
        raise RuntimeError("rebuild offline")

    def fail_vector(*args, **kwargs):
        raise OSError("id-map offline")

    monkeypatch.setattr(index_module, "save_id_map_json", fail_vector)
    coordinator = MutationCoordinator(
        settings_obj=runtime.settings, ports=replace(runtime.ports, reconcile_index=fail_rebuild)
    )
    batch = [
        MutationBatchItem(
            payload=PreparedMutation(
                intent=intent,
                vector_mode_enabled=True,
                precomputed_vectors_by_external_id={intent.upserts[0].external_id: [0.0, 1.0]},
            )
        )
        for intent in [_upsert("failed", "v2"), _upsert("next", "v2", "next")]
    ]
    coordinator._process_batch(batch=batch, journal=runtime.journal, use_atomic=False)
    assert all(isinstance(item.error, MutationRecoveryRequiredError) for item in batch)
    assert attempts == [True]
    assert runtime.journal.get("failed").state == "FAILED_NEEDS_RECOVERY"
    assert runtime.journal.get("next") is None
    assert [(d.external_id, d.content) for d in runtime.repo.get_all_documents()] == [("doc", "v1")]


def test_grouped_recovery_reconciles_once_and_keeps_terminals_pending_until_done(runtime):
    for name in ("a", "b"):
        runtime.repo.upsert_documents_by_external_id(
            [runtime.repo.UpsertDoc(external_id=name, content="v1")]
        )
        runtime.journal.upsert(
            MutationRecord(
                op_id=name,
                state="SQL_COMMITTED",
                intent=intent_to_dict(_upsert(name, external_id=name)),
                before_image={"docs": [], "existing_tombstones": []},
                vector_attempted=True,
            )
        )
    original = runtime.ports.reconcile_index

    def reconcile():
        assert runtime.repo.get_all_documents() == []
        assert {runtime.journal.get(name).state for name in ("a", "b")} == {"COMPENSATING"}
        return original()

    coordinator = MutationCoordinator(
        settings_obj=runtime.settings, ports=replace(runtime.ports, reconcile_index=reconcile)
    )
    assert coordinator.recover_incomplete() == 2
    assert len(runtime.rebuilds) == 1
    assert {runtime.journal.get(name).state for name in ("a", "b")} == {"ROLLED_BACK"}


def test_vector_committed_recovery_preserves_exact_outcome_and_record(runtime, monkeypatch):
    original = runtime.journal.upsert

    def fail_final_commit(record):
        if record.state == "COMMITTED":
            raise OSError("final journal write failed")
        return original(record)

    monkeypatch.setattr(runtime.journal, "upsert", fail_final_commit)
    intent = _upsert("commit-window")
    with pytest.raises(MutationRecoveryRequiredError, match="final journal"):
        runtime.coordinator.execute(intent)
    record = runtime.journal.get(intent.op_id)
    assert record.state == "VECTOR_COMMITTED"
    assert record.outcome["inserted"] == 1
    monkeypatch.setattr(runtime.journal, "upsert", original)
    assert runtime.coordinator.recover_incomplete() == 1
    assert runtime.journal.get(intent.op_id).outcome == record.outcome
    assert runtime.coordinator.execute(intent).inserted == 1
    assert runtime.journal.get(intent.op_id).state == "COMMITTED"
    assert runtime.rebuilds == []


def test_final_journal_write_failure_blocks_remaining_batch_items(runtime, monkeypatch):
    original_upsert = runtime.journal.upsert

    def fail_commit(record):
        if record.state == "COMMITTED":
            raise OSError("final journal offline")
        original_upsert(record)

    monkeypatch.setattr(runtime.journal, "upsert", fail_commit)
    batch = [
        MutationBatchItem(
            payload=runtime.coordinator._saga.prepare(
                intent=_upsert(name, external_id=name), vector_mode_enabled=True
            )
        )
        for name in ("first", "next")
    ]
    runtime.coordinator._process_batch(batch=batch, journal=runtime.journal, use_atomic=False)
    assert all(isinstance(item.error, MutationRecoveryRequiredError) for item in batch)
    assert runtime.journal.get("first").state == "VECTOR_COMMITTED"
    assert runtime.journal.get("next") is None
    assert [doc.external_id for doc in runtime.repo.get_all_documents()] == ["first"]
    assert runtime.rebuilds == []


@pytest.mark.parametrize("state", ["COMMITTED", "VECTOR_COMMITTED", "SQL_COMMITTED"])
@pytest.mark.parametrize("entrypoint", ["execute", "recover"])
def test_corrupt_record_preserves_journal_and_corpus(runtime, state, entrypoint):
    runtime.coordinator.execute(_upsert("initial"))
    path = runtime.journal.root / (hashlib.sha256(b"initial").hexdigest() + ".json")
    payload = json.loads(path.read_text())
    payload["state"] = state
    payload["before_image" if state == "SQL_COMMITTED" else "outcome"] = {}
    content = json.dumps(payload).encode()
    path.write_bytes(content)

    with pytest.raises(MutationRecoveryRequiredError) as error:
        if entrypoint == "execute":
            runtime.coordinator.execute(_upsert("new-op", "v2"))
        else:
            runtime.coordinator.recover_incomplete()

    assert str(path) in str(error.value)
    assert "initial" in str(error.value)
    assert state in str(error.value)
    assert path.read_bytes() == content
    assert [(d.external_id, d.content) for d in runtime.repo.get_all_documents()] == [("doc", "v1")]
    assert runtime.journal.get("new-op") is None
    assert runtime.rebuilds == []


@pytest.mark.parametrize("failed_state", ["ROLLED_BACK", "FAILED_NEEDS_RECOVERY"])
def test_recovery_journal_write_failure_remains_pending_and_typed(
    runtime, monkeypatch, failed_state
):
    runtime.coordinator.execute(_upsert("initial"))
    original_record = runtime.journal.get("initial")
    runtime.journal.upsert(replace(original_record, state="SQL_COMMITTED"))
    original_upsert = runtime.journal.upsert

    def fail_write(record):
        if record.state == failed_state:
            raise OSError("recovery journal offline")
        original_upsert(record)

    def fail_reconcile():
        raise OSError("rebuild offline")

    monkeypatch.setattr(runtime.journal, "upsert", fail_write)
    coordinator = MutationCoordinator(
        settings_obj=runtime.settings,
        ports=(
            replace(runtime.ports, reconcile_index=fail_reconcile)
            if failed_state == "FAILED_NEEDS_RECOVERY"
            else runtime.ports
        ),
    )
    with pytest.raises(MutationRecoveryRequiredError, match="recovery journal offline"):
        coordinator.execute(_upsert("blocked", "v2"))
    assert runtime.journal.get("initial").state == "COMPENSATING"
    assert runtime.journal.get("blocked") is None
    assert runtime.repo.get_all_documents() == []


def test_pending_recovery_blocks_next_operation_until_reconciled(runtime):
    runtime.coordinator.execute(_upsert("initial"))
    runtime.journal.upsert(replace(runtime.journal.get("initial"), state="SQL_COMMITTED"))

    def unavailable():
        raise OSError("rebuild offline")

    coordinator = MutationCoordinator(
        settings_obj=runtime.settings,
        ports=replace(runtime.ports, reconcile_index=unavailable),
    )
    with pytest.raises(MutationRecoveryRequiredError, match="rebuild offline"):
        coordinator.execute(_upsert("next", "v2"))
    assert runtime.journal.get("next") is None
    assert runtime.journal.get("initial").state == "FAILED_NEEDS_RECOVERY"

    result = runtime.coordinator.execute(_upsert("next", "v2"))
    assert result.inserted == 1
    assert runtime.journal.get("initial").state == "ROLLED_BACK"
    assert runtime.repo.get_all_documents()[0].content == "v2"
    assert runtime.vector().ntotal == 1
    assert len(runtime.rebuilds) == 1

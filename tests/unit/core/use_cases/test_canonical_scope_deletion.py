from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from local_rag_backend.core.services.maintenance import rebuild_index_from_db
from local_rag_backend.core.use_cases.docs_mutation import MutationCoordinator, MutationIntent
from local_rag_backend.infrastructure.persistence.shared.mutation_journal import FileMutationJournal
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage, base as db_base
from local_rag_backend.infrastructure.persistence.vector.storage import VectorStorage
from local_rag_backend.settings import Settings


@pytest.fixture()
def deletion_state(in_memory_sqlite, tmp_path):
    cfg = Settings(
        retrieval_mode="dense",
        vector_backend="numpy",
        data_dir=tmp_path,
        index_path=str(tmp_path / "index.npz"),
        id_map_path=str(tmp_path / "ids.json"),
    )
    repo = SqlDocumentStorage(session_factory=in_memory_sqlite)
    rows, _, _ = repo.upsert_documents_by_external_id(
        [repo.UpsertDoc(external_id=key, content=key, scope="demo") for key in ["keep", "stale"]]
    )
    vector = VectorStorage(
        cfg.index_path, cfg.id_map_path, dim=2, backend="numpy", settings_obj=cfg
    )
    vector.rebuild([row.id for row in rows], [[1.0, 0.0], [0.0, 1.0]])
    journal = FileMutationJournal(tmp_path / "journal")
    embedder = SimpleNamespace(embed=lambda texts: [[1.0, 0.0] for _ in texts])
    ports = SimpleNamespace(
        doc_repo_factory=lambda: repo,
        build_upsert_doc=repo.UpsertDoc,
        build_embedder=lambda: embedder,
        vector_repo_factory=lambda **kwargs: vector,
        reconcile_index=lambda: rebuild_index_from_db(
            doc_repo=repo, vec_repo=vector, embedder=embedder
        ),
        write_lock=lambda **kwargs: nullcontext(),
        mutation_journal_factory=lambda: journal,
        mutation_uow_factory=lambda: db_base.session_uow(session_factory=in_memory_sqlite),
        bump_rag_service_version=None,
    )
    return cfg, repo, vector, journal, ports


def test_scope_hard_delete_is_journaled_and_idempotent(deletion_state):
    cfg, repo, vector, journal, ports = deletion_state
    coordinator = MutationCoordinator(settings_obj=cfg, ports=ports)
    intent = MutationIntent(op_id="scope-delete", hard_delete_external_ids=("stale",))

    first = coordinator.execute(intent)
    replay = coordinator.execute(intent)

    assert first.deleted_sql == 1
    assert first.deleted_index == 1
    assert replay == first
    assert journal.get("scope-delete").state == "COMMITTED"
    assert {doc.external_id for doc in repo.get_all_documents()} == {"keep"}
    assert set(vector.vector_index.id_map) == {doc.id for doc in repo.get_all_documents()}
    assert not repo.get_tombstoned_external_ids(["stale"])


def test_failed_scope_vector_delete_restores_sql_and_index(deletion_state, monkeypatch):
    cfg, repo, vector, journal, ports = deletion_state
    coordinator = MutationCoordinator(settings_obj=cfg, ports=ports)

    def fail_delta(*, delete_ids, upserts):
        raise RuntimeError("injected vector failure")

    monkeypatch.setattr(vector, "apply_delta_atomic", fail_delta)
    with pytest.raises(RuntimeError, match="injected vector failure"):
        coordinator.execute(
            MutationIntent(op_id="scope-vector-failure", hard_delete_external_ids=("stale",))
        )

    assert journal.get("scope-vector-failure").state == "ROLLED_BACK"
    assert {doc.external_id for doc in repo.get_all_documents()} == {"keep", "stale"}
    assert set(vector.vector_index.id_map) == {doc.id for doc in repo.get_all_documents()}
    assert not repo.get_tombstoned_external_ids(["stale"])

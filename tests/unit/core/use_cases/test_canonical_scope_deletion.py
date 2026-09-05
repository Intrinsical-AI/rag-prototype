from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session, sessionmaker

from local_rag_backend.core.services.maintenance import rebuild_index_from_db
from local_rag_backend.core.use_cases.docs_import_canonical import _delete_stale_scope_documents
from local_rag_backend.core.use_cases.errors import IndexRebuildRequiredError
from local_rag_backend.core.use_cases.index import rebuild_index_sync
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage, base as db_base
from local_rag_backend.infrastructure.persistence.vector import index as vector_index_module
from local_rag_backend.infrastructure.persistence.vector.manifest import purge_index_artifacts
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
    repo = SqlDocumentStorage()
    rows, _, _ = repo.upsert_documents_by_external_id(
        [repo.UpsertDoc(external_id=key, content=key, scope="demo") for key in ["keep", "stale"]]
    )
    vector = VectorStorage(
        cfg.index_path, cfg.id_map_path, dim=2, backend="numpy", settings_obj=cfg
    )
    vector.rebuild([row.id for row in rows], [[1.0, 0.0], [0.0, 1.0]])
    ports = SimpleNamespace(
        doc_repo_factory=lambda: repo,
        vector_repo_factory=lambda **kwargs: vector,
        mutation_uow_factory=db_base.session_uow,
    )
    return cfg, repo, vector, ports


def _delete(state):
    cfg, _, _, ports = state
    return _delete_stale_scope_documents(
        scope="demo", keep_external_ids={"keep"}, settings_obj=cfg, ports=ports
    )


def _assert_sql_preserved(repo):
    # Each adapter read opens a new session after the failed UoW has exited.
    assert {doc.external_id for doc in repo.get_all_documents()} == {"keep", "stale"}
    assert not repo.get_tombstoned_external_ids(["stale"])


def test_scope_deletion_success_and_idempotence(deletion_state):
    _, repo, vector, _ = deletion_state
    assert _delete(deletion_state) == (["stale"], 1, 1)
    assert {doc.external_id for doc in repo.get_all_documents()} == {"keep"}
    assert set(vector.vector_index.id_map) == {doc.id for doc in repo.get_all_documents()}
    assert _delete(deletion_state) == ([], 0, 0)
    assert not repo.get_tombstoned_external_ids(["stale"])


@pytest.mark.parametrize(
    "missing", ["mutation_uow_factory", "snapshot_by_external_ids", "apply_delta_atomic"]
)
def test_scope_deletion_preflights_capabilities(deletion_state, monkeypatch, missing):
    _, repo, vector, ports = deletion_state
    target = (
        ports
        if missing == "mutation_uow_factory"
        else vector
        if missing == "apply_delta_atomic"
        else repo
    )
    monkeypatch.setattr(target, missing, None)
    with pytest.raises(RuntimeError):
        _delete(deletion_state)
    _assert_sql_preserved(repo)
    assert vector.ntotal == 2


def test_sql_delete_failure_does_not_apply_vector_delta(deletion_state, monkeypatch):
    _, repo, vector, _ = deletion_state
    original_delete = repo.hard_delete_by_external_ids

    def fail_after_sql(ids):
        original_delete(ids)
        raise RuntimeError("SQL delete failed")

    monkeypatch.setattr(repo, "hard_delete_by_external_ids", fail_after_sql)
    with pytest.raises(RuntimeError, match="SQL delete failed"):
        _delete(deletion_state)
    _assert_sql_preserved(repo)
    assert vector.ntotal == 2


@pytest.mark.parametrize("failure", ["before_vector_save", "between_vector_files", "sql_commit"])
def test_failed_delta_or_final_commit_rolls_back_sql_and_can_rebuild(
    deletion_state, monkeypatch, failure, in_memory_sqlite
):
    cfg, repo, vector, ports = deletion_state

    def fail(*args, **kwargs):
        raise RuntimeError("injected write failure")

    with monkeypatch.context() as fault:
        if failure == "before_vector_save":
            fault.setattr(vector.vector_index.engine, "save", fail)
        elif failure == "between_vector_files":
            fault.setattr(vector_index_module, "save_id_map_json", fail)
        else:

            class FailingCommitSession(Session):
                pass

            event.listen(FailingCommitSession, "before_commit", fail)
            factory = sessionmaker(bind=db_base.engine, class_=FailingCommitSession)
            fault.setattr(
                ports, "mutation_uow_factory", lambda: db_base.session_uow(session_factory=factory)
            )
        with pytest.raises(IndexRebuildRequiredError, match="rag-rebuild-index"):
            _delete(deletion_state)
    _assert_sql_preserved(repo)
    if failure == "sql_commit":
        assert vector.ntotal == 1
    elif failure == "between_vector_files":
        with pytest.raises(RuntimeError, match="length mismatch"):
            VectorStorage(cfg.index_path, cfg.id_map_path, dim=2, backend="numpy", settings_obj=cfg)

    rebuilt = []

    def new_vector(**kwargs):
        result = VectorStorage(**kwargs)
        rebuilt.append(result)
        return result

    # Exercise the same use case used by rag-rebuild-index, including artifact cleanup.
    assert (
        rebuild_index_sync(
            settings_obj=cfg,
            ports=SimpleNamespace(
                doc_repo_factory=lambda: repo,
                build_embedder=lambda: SimpleNamespace(
                    dim=2, embed=lambda texts: [[1.0, 0.0] for _ in texts]
                ),
                purge_index_artifacts_fn=purge_index_artifacts,
                vector_repo_factory=new_vector,
                rebuild_fn=rebuild_index_from_db,
            ),
        )
        == 2
    )
    assert set(rebuilt[0].vector_index.id_map) == {doc.id for doc in repo.get_all_documents()}
    assert _delete_stale_scope_documents(
        scope="demo",
        keep_external_ids={"keep"},
        settings_obj=cfg,
        ports=SimpleNamespace(
            doc_repo_factory=lambda: repo,
            vector_repo_factory=lambda **kwargs: rebuilt[0],
            mutation_uow_factory=db_base.session_uow,
        ),
    ) == (["stale"], 1, 1)

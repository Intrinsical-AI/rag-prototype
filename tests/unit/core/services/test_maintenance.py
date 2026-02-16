import pytest

from local_rag_backend.core.services.maintenance import (
    delete_documents_multi_store,
    rebuild_index_from_db,
)


class _Doc:
    def __init__(self, id: int, content: str) -> None:
        self.id = id
        self.content = content


class DummyDocRepo:
    def __init__(self, docs):
        self._docs = list(docs)
        self.deleted = []

    def get_all_documents(self):
        return list(self._docs)

    def get(self, ids):
        s = set(ids)
        return [d for d in self._docs if d.id in s]

    def delete_documents(self, ids):
        s = set(ids)
        self.deleted.extend(list(ids))
        self._docs = [d for d in self._docs if d.id not in s]


class DummyEmbedder:
    def __init__(self, dim=2) -> None:
        self.dim = dim
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t)), 0.0] for t in texts]


class DummyVecRepo:
    def __init__(self, fail_delete=False, fail_rebuild=False) -> None:
        self.rebuild_calls = []
        self.delete_calls = []
        self.fail_delete = fail_delete
        self.fail_rebuild = fail_rebuild

    def rebuild(self, ids, vectors):
        if self.fail_rebuild:
            raise RuntimeError("fail-rebuild")
        self.rebuild_calls.append((list(ids), list(vectors)))

    def delete(self, ids):
        self.delete_calls.append(list(ids))
        if self.fail_delete:
            raise RuntimeError("fail-delete")
        return len(list(ids))


def test_rebuild_index_from_db_empty_rebuilds_to_empty():
    doc_repo = DummyDocRepo([])
    vec_repo = DummyVecRepo()
    embedder = DummyEmbedder()

    n = rebuild_index_from_db(doc_repo=doc_repo, vec_repo=vec_repo, embedder=embedder)
    assert n == 0
    assert vec_repo.rebuild_calls == [([], [])]


def test_delete_documents_sql_only_counts_existing():
    doc_repo = DummyDocRepo([_Doc(1, "a"), _Doc(2, "b")])
    deleted_sql, deleted_index, rebuilt = delete_documents_multi_store(
        doc_repo=doc_repo, ids=[2, 999]
    )
    assert deleted_sql == 1
    assert deleted_index is None
    assert rebuilt is False
    assert [d.id for d in doc_repo.get_all_documents()] == [1]


def test_delete_documents_index_failure_triggers_rebuild():
    doc_repo = DummyDocRepo([_Doc(1, "a"), _Doc(2, "bb")])
    vec_repo = DummyVecRepo(fail_delete=True)
    embedder = DummyEmbedder()

    deleted_sql, deleted_index, rebuilt = delete_documents_multi_store(
        doc_repo=doc_repo,
        vec_repo=vec_repo,
        embedder=embedder,
        ids=[1],
        rebuild_on_index_failure=True,
    )
    assert deleted_sql == 1
    assert deleted_index is None
    assert rebuilt is True
    assert vec_repo.rebuild_calls  # rebuilt from DB state (doc 2 remains)


def test_delete_documents_does_not_build_embedder_when_index_delete_succeeds():
    doc_repo = DummyDocRepo([_Doc(1, "a"), _Doc(2, "bb")])
    vec_repo = DummyVecRepo(fail_delete=False)
    factory_calls = 0

    def _embedder_factory():
        nonlocal factory_calls
        factory_calls += 1
        return DummyEmbedder()

    deleted_sql, deleted_index, rebuilt = delete_documents_multi_store(
        doc_repo=doc_repo,
        vec_repo=vec_repo,
        embedder_factory=_embedder_factory,
        ids=[1],
        rebuild_on_index_failure=True,
    )
    assert deleted_sql == 1
    assert deleted_index == 1
    assert rebuilt is False
    assert factory_calls == 0


def test_delete_documents_uses_real_deleted_index_count():
    doc_repo = DummyDocRepo([_Doc(1, "a"), _Doc(2, "bb"), _Doc(3, "ccc")])

    class PartialDeleteVec(DummyVecRepo):
        def delete(self, ids):
            super().delete(ids)
            return 1

    vec_repo = PartialDeleteVec()
    deleted_sql, deleted_index, rebuilt = delete_documents_multi_store(
        doc_repo=doc_repo,
        vec_repo=vec_repo,
        embedder=DummyEmbedder(),
        ids=[1, 2],
        rebuild_on_index_failure=True,
    )
    assert deleted_sql == 2
    assert deleted_index == 1
    assert rebuilt is False


def test_delete_documents_index_failure_no_rebuild_raises():
    doc_repo = DummyDocRepo([_Doc(1, "a")])
    vec_repo = DummyVecRepo(fail_delete=True)

    with pytest.raises(RuntimeError, match="fail-delete"):
        delete_documents_multi_store(
            doc_repo=doc_repo,
            vec_repo=vec_repo,
            embedder=None,
            ids=[1],
            rebuild_on_index_failure=False,
        )


def test_delete_documents_index_and_rebuild_failure_raises_consistency_error():
    doc_repo = DummyDocRepo([_Doc(1, "a"), _Doc(2, "b")])
    vec_repo = DummyVecRepo(fail_delete=True, fail_rebuild=True)
    embedder = DummyEmbedder()

    with pytest.raises(RuntimeError, match="Multi-store inconsistency risk"):
        delete_documents_multi_store(
            doc_repo=doc_repo,
            vec_repo=vec_repo,
            embedder=embedder,
            ids=[1],
            rebuild_on_index_failure=True,
        )


def test_delete_documents_aborts_before_sql_when_vector_preflight_fails_without_embedder():
    doc_repo = DummyDocRepo([_Doc(1, "a"), _Doc(2, "b")])

    class PreflightFailVec(DummyVecRepo):
        def delete(self, ids):
            self.delete_calls.append(list(ids))
            if not ids:
                raise RuntimeError("manifest-drift")
            return len(list(ids))

    vec_repo = PreflightFailVec()

    def _embedder_factory():
        raise RuntimeError("embedder-unavailable")

    with pytest.raises(RuntimeError, match="Aborting SQL delete"):
        delete_documents_multi_store(
            doc_repo=doc_repo,
            vec_repo=vec_repo,
            embedder_factory=_embedder_factory,
            ids=[1],
            rebuild_on_index_failure=True,
        )

    # Critical: SQL state must remain untouched when we cannot guarantee dense repair.
    assert [d.id for d in doc_repo.get_all_documents()] == [1, 2]
    assert doc_repo.deleted == []


def test_delete_documents_raises_explicit_consistency_error_when_fallback_embedder_unavailable():
    doc_repo = DummyDocRepo([_Doc(1, "a"), _Doc(2, "b")])

    class DeleteOnlyFailVec(DummyVecRepo):
        def delete(self, ids):
            self.delete_calls.append(list(ids))
            if ids:
                raise RuntimeError("fail-delete")
            return 0

    vec_repo = DeleteOnlyFailVec()

    def _embedder_factory():
        raise RuntimeError("embedder-unavailable")

    with pytest.raises(RuntimeError, match="Multi-store inconsistency risk"):
        delete_documents_multi_store(
            doc_repo=doc_repo,
            vec_repo=vec_repo,
            embedder_factory=_embedder_factory,
            ids=[1],
            rebuild_on_index_failure=True,
        )

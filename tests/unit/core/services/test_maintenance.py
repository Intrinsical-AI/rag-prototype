from types import SimpleNamespace

import pytest

from local_rag_backend.core.services.maintenance import rebuild_index_from_db


class _VectorRepo:
    def __init__(self):
        self.calls = []

    def rebuild(self, ids, vectors):
        self.calls.append((list(ids), list(vectors)))


def test_rebuild_empty_clears_old_index_without_embedding():
    vec = _VectorRepo()
    repo = SimpleNamespace(get_all_documents=lambda: [])
    assert rebuild_index_from_db(doc_repo=repo, vec_repo=vec, embedder=None) == 0
    assert vec.calls == [([], [])]


def test_rebuild_embeds_batches_and_validates_count_before_write():
    docs = [SimpleNamespace(id=str(i), content=str(i)) for i in range(3)]
    repo = SimpleNamespace(get_all_documents=lambda: docs)
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return [[float(text)] for text in texts]

    vec = _VectorRepo()
    assert (
        rebuild_index_from_db(
            doc_repo=repo, vec_repo=vec, embedder=SimpleNamespace(embed=embed), batch_size=2
        )
        == 3
    )
    assert calls == [["0", "1"], ["2"]]
    assert vec.calls == [(["0", "1", "2"], [[0.0], [1.0], [2.0]])]
    with pytest.raises(ValueError, match="1 vectors for 3 docs"):
        rebuild_index_from_db(
            doc_repo=repo, vec_repo=vec, embedder=SimpleNamespace(embed=lambda texts: [[1.0]])
        )
    assert len(vec.calls) == 1

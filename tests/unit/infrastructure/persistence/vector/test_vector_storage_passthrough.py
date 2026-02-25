import numpy as np
import pytest

from local_rag_backend.infrastructure.persistence.vector.storage import VectorStorage


class _DummyVectorIndex:
    def __init__(self, index_path, id_map_path, dim, **_kwargs):
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.dim = dim
        self.add_calls = []

    def add_to_index(self, ids, vecs):
        self.add_calls.append((ids, vecs))

    def search(self, query_vector, k):
        return np.asarray([1, 2], dtype=np.int64), np.asarray([0.1, 0.2], dtype=np.float32)


def test_vector_storage_upsert_calls_index(monkeypatch):
    monkeypatch.setattr(
        "local_rag_backend.infrastructure.persistence.vector.storage.VectorIndex",
        _DummyVectorIndex,
        raising=True,
    )
    st = VectorStorage(index_path="i", id_map_path="m", dim=3)
    st.upsert([10, 11], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert len(st.vector_index.add_calls) == 1
    ids, vecs = st.vector_index.add_calls[0]
    assert ids == [10, 11]
    assert vecs == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


def test_vector_storage_search_passthrough(monkeypatch):
    monkeypatch.setattr(
        "local_rag_backend.infrastructure.persistence.vector.storage.VectorIndex",
        _DummyVectorIndex,
        raising=True,
    )
    st = VectorStorage(index_path="i", id_map_path="m", dim=3)
    idxs, dists = st.search([0.0, 0.0, 0.0], k=2)
    assert idxs.tolist() == [1, 2]
    assert dists.tolist() == pytest.approx([0.1, 0.2])

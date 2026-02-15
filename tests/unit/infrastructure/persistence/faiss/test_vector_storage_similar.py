# tests/unit/infrastructure/persistence/faiss/test_vector_storage_similar.py
import numpy as np

from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage


class DummyFaissIndex:
    def __init__(self):
        self.id_map = [101, 102, 103]
        # tres dists: 0.0 (idéntico), 1.0, 9.0

    def search(self, q, k):
        return np.array([0, 1, 2]), np.array([0.0, 1.0, 9.0])


def test_similar_normalizes_and_maps_ids(monkeypatch):
    # inyectar dummy
    monkeypatch.setattr(
        "local_rag_backend.infrastructure.persistence.faiss.faiss_.FaissIndex",
        lambda *a, **k: None,
        raising=True,
    )
    storage = FaissVectorStorage(index_path=":mem:", id_map_path=":mem:", dim=3)
    storage.faiss_index = DummyFaissIndex()

    pairs = storage.similar([0.0, 0.0, 0.0], k=3)
    ids, sims = zip(*pairs, strict=False)
    assert list(ids) == [101, 102, 103]
    # sim(0.0) > sim(1.0) > sim(9.0) y todo en [0,1]
    assert 0.0 <= min(sims) <= max(sims) <= 1.0
    assert sims[0] > sims[1] > sims[2]


def test_similar_is_fail_safe_on_id_map_mismatch(monkeypatch):
    class DummyMismatchIndex:
        id_map: list[int] = []  # mismatch: search returns indices not present in id_map

        def search(self, q, k):
            return np.array([0]), np.array([0.0])

    monkeypatch.setattr(
        "local_rag_backend.infrastructure.persistence.faiss.faiss_.FaissIndex",
        lambda *a, **k: None,
        raising=True,
    )
    storage = FaissVectorStorage(index_path=":mem:", id_map_path=":mem:", dim=3)
    storage.faiss_index = DummyMismatchIndex()

    # Should not raise IndexError.
    assert storage.similar([0.0, 0.0, 0.0], k=1) == []

# tests/unit/infrastructure/persistence/vector/test_vector_storage_similar.py
import numpy as np

from local_rag_backend.infrastructure.persistence.vector.storage import VectorStorage


class DummyVectorIndex:
    def __init__(self):
        self.id_map = [101, 102, 103]
        # tres dists: 0.0 (idéntico), 1.0, 9.0

    def search(self, q, k):
        return np.array([0, 1, 2]), np.array([0.0, 1.0, 9.0])


def test_similar_normalizes_and_maps_ids(monkeypatch):
    # inyectar dummy
    monkeypatch.setattr(
        "local_rag_backend.infrastructure.persistence.vector.storage.VectorIndex",
        lambda *a, **k: None,
        raising=True,
    )
    storage = VectorStorage(index_path=":mem:", id_map_path=":mem:", dim=3)
    storage.vector_index = DummyVectorIndex()

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
        "local_rag_backend.infrastructure.persistence.vector.storage.VectorIndex",
        lambda *a, **k: None,
        raising=True,
    )
    storage = VectorStorage(index_path=":mem:", id_map_path=":mem:", dim=3)
    storage.vector_index = DummyMismatchIndex()

    # Should not raise IndexError.
    assert storage.similar([0.0, 0.0, 0.0], k=1) == []


def test_similar_prefers_atomic_search_snapshot_when_available(monkeypatch):
    class SnapshotAwareIndex:
        def __init__(self):
            self.id_map = [999, 102]

        def search_with_snapshot(self, q, k):
            _ = q, k
            # Return indices + an immutable snapshot captured under lock.
            return np.array([0]), np.array([0.0]), [101, 102]

    monkeypatch.setattr(
        "local_rag_backend.infrastructure.persistence.vector.storage.VectorIndex",
        lambda *a, **k: None,
        raising=True,
    )
    storage = VectorStorage(index_path=":mem:", id_map_path=":mem:", dim=3)
    storage.vector_index = SnapshotAwareIndex()

    pairs = storage.similar([0.0, 0.0, 0.0], k=1)
    assert pairs == [(101, 1.0)]

import numpy as np

from local_rag_backend.infrastructure.persistence.vector.storage import VectorStorage
from local_rag_backend.settings import Settings


def _storage(tmp_path):
    store = VectorStorage(
        index_path=str(tmp_path / "index.npy"),
        id_map_path=str(tmp_path / "ids.json"),
        dim=3,
        backend="numpy",
        settings_obj=Settings(),
    )
    store.rebuild(["101", "102", "103"], [[0, 0, 0], [1, 0, 0], [3, 0, 0]])
    return store


def test_similar_normalizes_and_maps_ids(tmp_path):
    pairs = _storage(tmp_path).similar([0.0, 0.0, 0.0], k=3)
    ids, sims = zip(*pairs, strict=True)
    assert list(ids) == ["101", "102", "103"]
    assert 0.0 <= min(sims) <= max(sims) <= 1.0
    assert sims[0] > sims[1] > sims[2]


def test_similar_falls_back_to_zero_for_flat_scores(tmp_path):
    store = _storage(tmp_path)
    store.rebuild(["101", "102"], [[0, 0, 0], [0, 0, 0]])
    assert store.similar([0.0, 0.0, 0.0], k=2) == [("101", 0.0), ("102", 0.0)]


def test_similar_is_fail_safe_on_id_map_mismatch(tmp_path, monkeypatch):
    store = _storage(tmp_path)
    monkeypatch.setattr(
        store.vector_index,
        "search_with_snapshot",
        lambda _query, _k: (np.array([0]), np.array([0.0]), []),
    )
    assert store.similar([0.0, 0.0, 0.0], k=1) == []


def test_similar_uses_atomic_search_snapshot(tmp_path, monkeypatch):
    store = _storage(tmp_path)
    store.vector_index.id_map = ["999", "102"]
    monkeypatch.setattr(
        store.vector_index,
        "search_with_snapshot",
        lambda _query, _k: (np.array([0]), np.array([0.0]), ["101", "102"]),
    )
    assert store.similar([0.0, 0.0, 0.0], k=1) == [("101", 1.0)]

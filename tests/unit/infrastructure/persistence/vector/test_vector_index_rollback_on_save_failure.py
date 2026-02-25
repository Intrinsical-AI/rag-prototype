import pytest

from local_rag_backend.infrastructure.persistence.vector.index import VectorIndex


def test_add_to_index_rolls_back_in_memory_state_on_save_failure(tmp_path, monkeypatch):
    """
    If persisting the index/id_map fails after mutating the in-memory index,
    the process must revert to the last known on-disk state. Otherwise retrieval
    can return IDs that were never committed (or were rolled back) until restart.
    """
    idx_path = tmp_path / "idx.npy"
    map_path = tmp_path / "id_map.json"

    idx = VectorIndex(idx_path, map_path, dim=2)
    idx.add_to_index([1], [[0.0, 0.0]])
    assert idx.id_map == [1]

    def _boom() -> None:
        raise RuntimeError("boom-save")

    monkeypatch.setattr(idx, "_save_locked", _boom, raising=True)

    with pytest.raises(RuntimeError, match="boom-save"):
        idx.add_to_index([2], [[1.0, 1.0]])

    # In-memory state should have been reloaded to the on-disk last good state.
    assert idx.id_map == [1]

    # And a fresh reload should also match.
    idx2 = VectorIndex(idx_path, map_path, dim=2)
    assert idx2.id_map == [1]

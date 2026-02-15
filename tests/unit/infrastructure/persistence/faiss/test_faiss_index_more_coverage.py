import json

import pytest

from local_rag_backend.infrastructure.persistence.faiss.index import FaissIndex


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(_faiss_available(), reason="These tests target the numpy fallback path")
def test_dim_none_requires_existing_index_file(tmp_path):
    with pytest.raises(ValueError, match="dim is required"):
        FaissIndex(tmp_path / "missing.index", tmp_path / "m.json", dim=None)


@pytest.mark.skipif(_faiss_available(), reason="These tests target the numpy fallback path")
def test_rebuild_empty_persists_and_loads(tmp_path):
    index_path = tmp_path / "idx.bin"
    id_map_path = tmp_path / "id_map.json"

    fi = FaissIndex(index_path, id_map_path, dim=3)
    fi.rebuild([], [])

    assert json.loads(id_map_path.read_text(encoding="utf-8")) == []
    fi2 = FaissIndex(index_path, id_map_path, dim=3)
    assert fi2.id_map == []


@pytest.mark.skipif(_faiss_available(), reason="These tests target the numpy fallback path")
def test_delete_ids_removes_all_occurrences(tmp_path):
    index_path = tmp_path / "idx.bin"
    id_map_path = tmp_path / "id_map.json"

    fi = FaissIndex(index_path, id_map_path, dim=2)
    fi.add_to_index([1, 2, 2, 3], [[0.0, 0.0], [10.0, 0.0], [10.0, 0.0], [0.0, 10.0]])

    deleted = fi.delete_ids([2])
    assert deleted == 2
    assert 2 not in fi.id_map


@pytest.mark.skipif(_faiss_available(), reason="These tests target the numpy fallback path")
def test_rebuild_length_mismatch_raises(tmp_path):
    fi = FaissIndex(tmp_path / "idx.bin", tmp_path / "m.json", dim=2)
    with pytest.raises(ValueError, match="length mismatch"):
        fi.rebuild([1], [[0.0, 0.0], [1.0, 1.0]])

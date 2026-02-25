import pickle

import numpy as np
import pytest

from local_rag_backend.infrastructure.persistence.vector.index import VectorIndex


def test_add_to_index_raises_on_len_mismatch(tmp_path):
    idx = VectorIndex(tmp_path / "i.faiss", tmp_path / "m.json", dim=4)
    with pytest.raises(ValueError, match="length mismatch"):
        idx.add_to_index([1, 2], [np.ones(4, dtype="float32")])


def test_invalid_id_map_format_raises(tmp_path):
    index_path = tmp_path / "i.faiss"
    id_map_path = tmp_path / "m.pkl"
    with id_map_path.open("wb") as f:
        pickle.dump([1, 2, 3], f)

    with pytest.raises(RuntimeError, match="Unsafe pickle id_map"):
        VectorIndex(index_path, id_map_path, dim=4)


def test_invalid_json_id_map_format_raises(tmp_path):
    index_path = tmp_path / "i.faiss"
    id_map_path = tmp_path / "m.json"
    id_map_path.write_text('{"not": "a list"}', encoding="utf-8")

    with pytest.raises(ValueError, match="id_map"):
        VectorIndex(index_path, id_map_path, dim=4)

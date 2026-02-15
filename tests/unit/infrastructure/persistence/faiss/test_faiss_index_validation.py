import pickle

import numpy as np
import pytest

from local_rag_backend.infrastructure.persistence.faiss.index import FaissIndex


def test_add_to_index_raises_on_len_mismatch(tmp_path):
    idx = FaissIndex(tmp_path / "i.faiss", tmp_path / "m.pkl", dim=4)
    with pytest.raises(ValueError, match="length mismatch"):
        idx.add_to_index([1, 2], [np.ones(4, dtype="float32")])


def test_invalid_id_map_format_raises(tmp_path):
    index_path = tmp_path / "i.faiss"
    id_map_path = tmp_path / "m.pkl"
    with id_map_path.open("wb") as f:
        pickle.dump({"not": "a list"}, f)

    with pytest.raises(ValueError, match="id_map"):
        FaissIndex(index_path, id_map_path, dim=4)

# tests/unit/infrastructure/persistence/vector/test_vector_index_dim_mismatch.py
import numpy as np
import pytest

from local_rag_backend.infrastructure.persistence.vector.index import VectorIndex


def test_add_to_index_raises_on_dim_mismatch(tmp_path):
    idx = VectorIndex(tmp_path / "i.faiss", tmp_path / "m.json", dim=4)
    with pytest.raises(ValueError, match="FAISS dim mismatch"):
        idx.add_to_index([1], [np.ones(3, dtype="float32")])  # 3 != 4

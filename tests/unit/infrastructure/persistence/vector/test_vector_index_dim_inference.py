import json

import numpy as np

from local_rag_backend.infrastructure.persistence.vector.index import VectorIndex


def test_dim_inference_from_numpy_index(tmp_path):
    index_path = tmp_path / "idx.npy"
    id_map_path = tmp_path / "id_map.json"

    # Write a numpy fallback "index" (vectors) directly.
    vectors = np.zeros((3, 7), dtype="float32")
    with index_path.open("wb") as f:
        np.save(f, vectors, allow_pickle=False)
    id_map_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    idx = VectorIndex(index_path, id_map_path, dim=None)
    assert idx.dim == 7
    assert idx.id_map == [1, 2, 3]

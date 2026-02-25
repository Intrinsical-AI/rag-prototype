# tests/test_vector_index.py
import numpy as np
from pytest import approx

from local_rag_backend.infrastructure.persistence.vector.index import VectorIndex


def test_faiss_add_and_search(tmp_path):
    dim = 4
    index_file = tmp_path / "test.faiss"
    idmap_file = tmp_path / "id_map.json"

    fi = VectorIndex(index_file, idmap_file, dim=dim)

    # We create 5 vectors – the first one is clearly different (all zeros)
    vecs = [np.zeros(dim, dtype="float32")]
    vecs += [np.random.rand(dim).astype("float32") for _ in range(4)]
    ids = [10, 11, 12, 13, 14]

    fi.add_to_index(ids, vecs)

    idxs, dists = fi.search(vecs[0], k=3)

    # The first result should be the identical vector (distance 0)
    assert idxs[0] != -1
    top_id = fi.id_map[idxs[0]]
    assert top_id == 10
    assert dists[0] == approx(0.0)

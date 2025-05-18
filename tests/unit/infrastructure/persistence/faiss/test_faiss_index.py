# tests/test_faiss_index.py
import numpy as np
from pytest import approx

from src.infrastructure.persistence.faiss.index import FaissIndex

"""
Revisar locks si el proycto crece
"""


def test_faiss_add_and_search(tmp_path):
    dim = 4
    index_file = tmp_path / "test.faiss"
    idmap_file = tmp_path / "id_map.pkl"

    fi = FaissIndex(index_file, idmap_file, dim=dim)

    # Creamos 5 vectores – el primero es claramente distinto (todo ceros)
    vecs = [np.zeros(dim, dtype="float32")]
    vecs += [np.random.rand(dim).astype("float32") for _ in range(4)]
    ids = [10, 11, 12, 13, 14]

    fi.add_to_index(ids, vecs)

    idxs, dists = fi.search(vecs[0], k=3)

    # El primer resultado debe ser el vector idéntico (distancia 0)
    assert idxs[0] != -1
    top_id = fi.id_map[idxs[0]]
    assert top_id == 10
    assert dists[0] == approx(0.0)

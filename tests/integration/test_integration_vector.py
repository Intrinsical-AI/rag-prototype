import numpy as np

from local_rag_backend.infrastructure.persistence.vector.index import VectorIndex


def test_vector_index_add_and_search(tmp_path):
    index_path = tmp_path / "test.index"
    id_map_path = tmp_path / "test_map.json"
    dim = 4
    idx = VectorIndex(index_path, id_map_path, dim=dim)
    vectors = [np.random.rand(dim).astype(np.float32) for _ in range(3)]
    ids = ["doc:101", "doc:102", "doc:103"]
    idx.add_to_index(ids, vectors)
    # Save and reload (persists well)
    idx2 = VectorIndex(index_path, id_map_path, dim=dim)
    q = vectors[0]
    idxs, _dists = idx2.search(q, k=1)
    assert idx2.id_map[idxs[0]] == ids[0]

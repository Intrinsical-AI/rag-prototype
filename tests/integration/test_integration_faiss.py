import numpy as np

from src.infrastructure.persistence.faiss.index import FaissIndex


def test_faiss_index_add_and_search(tmp_path):
    index_path = tmp_path / "test.index"
    id_map_path = tmp_path / "test_map.pkl"
    dim = 4
    idx = FaissIndex(index_path, id_map_path, dim=dim)
    vectors = [np.random.rand(dim).astype(np.float32) for _ in range(3)]
    ids = [101, 102, 103]
    idx.add_to_index(ids, vectors)
    # Guardar y recargar (persiste bien)
    idx2 = FaissIndex(index_path, id_map_path, dim=dim)
    q = vectors[0]
    idxs, dists = idx2.search(q, k=1)
    assert idx2.id_map[idxs[0]] == ids[0]

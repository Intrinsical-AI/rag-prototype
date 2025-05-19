# src/adapters/storage/faiss_crud.py

from typing import Sequence

from src.core.ports import VectorRepoPort
from src.infrastructure.persistence.faiss.index import FaissIndex


class FaissVectorStorage(VectorRepoPort):
    """
    Adapter que implementa VectorRepoPort usando FAISS.
    """

    def __init__(self, index_path: str, id_map_path: str):
        self.faiss_index = FaissIndex(index_path, id_map_path)

    def upsert(self, ids: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
        self.faiss_index.add_to_index(list(ids), list(vectors))

    def similar(self, vector, k: int):
        idxs, dists = self.faiss_index.search(vector, k)
        real_ids = [self.faiss_index.id_map[i] for i in idxs if i != -1]
        return list(zip(real_ids, dists))


"""
faiss = DenseFaissRetriever(embedder=embedder, doc_repo=sql_repo, ...)
retriever = IdMapperRetriever(faiss, sql_repo)
"""

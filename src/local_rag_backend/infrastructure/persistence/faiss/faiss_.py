# src/infrastructure/persistence/faiss/faiss_.py

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from local_rag_backend.core.ports import VectorRepoPort
from local_rag_backend.infrastructure.persistence.faiss.index import FaissIndex
from local_rag_backend.utils import normalize_similarities_from_distances


class FaissVectorStorage(VectorRepoPort):
    """
    Adapter que implementa VectorRepoPort usando FAISS.
    """

    def __init__(self, index_path: str, id_map_path: str, dim: int | None = None):
        self.faiss_index = FaissIndex(index_path, id_map_path, dim=dim or 384)

    def upsert(self, ids: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
        self.faiss_index.add_to_index(list(ids), list(vectors))

    def search(
        self, query_vector: Sequence[float], k: int
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        return self.faiss_index.search(query_vector, k)

    @property
    def id_map(self) -> list[int]:
        return self.faiss_index.id_map

    def similar(self, vector: Sequence[float], k: int) -> list[tuple[int, float]]:
        idxs, dists = self.search(vector, k)
        # Convert L2 distances to similarities and normalize to [0,1]
        # Filter out invalid entries (-1) before normalization
        valid_pairs = [(i, d) for i, d in zip(idxs, dists, strict=False) if i != -1]
        if valid_pairs:
            valid_dists = [float(d) for _, d in valid_pairs]
            sims = normalize_similarities_from_distances(valid_dists)
        else:
            sims = []
        pairs: list[tuple[int, float]] = []
        for (i, _), sim in zip(valid_pairs, sims, strict=False):
            real_id = self.faiss_index.id_map[i]
            pairs.append((real_id, float(sim)))
        return pairs


"""
faiss = DenseFaissRetriever(embedder=embedder, doc_repo=sql_repo, ...)
retriever = IdMapperRetriever(faiss, sql_repo)
"""

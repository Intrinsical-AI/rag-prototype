# src/infrastructure/persistence/faiss/faiss_.py
"""
Adapter for vector storage and search using FAISS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_backend.core.ports import VectorRepoPort
from local_rag_backend.infrastructure.persistence.faiss.index import FaissIndex

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray


class FaissVectorStorage(VectorRepoPort):
    """Adapter for vector storage and search using FAISS."""

    def __init__(self, index_path: str, id_map_path: str, dim: int = 384):
        self.faiss_index = FaissIndex(index_path, id_map_path, dim)

    def upsert(self, ids: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
        """Add vectors to the FAISS index."""
        self.faiss_index.add_to_index(list(ids), list(vectors))

    def search(
        self, query_vector: Sequence[float], k: int
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Perform a raw search in the FAISS index."""
        return self.faiss_index.search(query_vector, k)

    def similar(self, vector: Sequence[float], k: int) -> list[tuple[int, float]]:
        """Find similar items and return their IDs and normalized similarity scores."""
        indices, distances = self.search(vector, k)

        # Filter out invalid indices (-1)
        valid_results = [
            (self.faiss_index.id_map[i], float(d))
            for i, d in zip(indices, distances, strict=False)
            if i != -1
        ]
        if not valid_results:
            return []

        # Normalize distances to similarity scores [0, 1]
        doc_ids, valid_distances = zip(*valid_results, strict=False)
        sim_raw = [1.0 / (1.0 + d) for d in valid_distances]
        min_s, max_s = min(sim_raw), max(sim_raw)

        if max_s == min_s:
            normalized_sims = [1.0] * len(sim_raw)
        else:
            normalized_sims = [(s - min_s) / (max_s - min_s) for s in sim_raw]

        return list(zip(doc_ids, normalized_sims, strict=False))

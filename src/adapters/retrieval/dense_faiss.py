from __future__ import annotations

"""Dense FAISS Retriever.

++ 

* Accept injectable `embedder` to avoid downloads during tests.
* Support both list and dict formats for `id_map` for backward compatibility.
* Ensure dimension consistency between FAISS index and embedder.
* Public API: `retrieve(query: str, k: int = 5)` → `(ids, scores)`.
"""

import pickle
from pathlib import Path
from typing import List, Tuple, Sequence

import faiss  # type: ignore
import numpy as np

from src.core.ports import RetrieverPort, EmbedderPort
from src.settings import settings
from src.adapters.embeddings.sentence_transformers import (
    SentenceTransformerEmbedder,
)

__all__ = ["DenseFaissRetriever"]


class DenseFaissRetriever(RetrieverPort):
    """Embedding-based search using FAISS.

    Parameters
    ----------
    embedder : EmbedderPort, optional
        Implementation of `EmbedderPort` to convert text into embeddings.
        Defaults to `SentenceTransformerEmbedder` if None.
    index_path : str | Path, optional
        Path to the FAISS index file. Default obtained from `settings`.
    id_map_path : str | Path, optional
        Path to the ID mapping file. Default obtained from `settings`.
    """

    def __init__(
        self,
        *,
        embedder: EmbedderPort | None = None,
        index_path: str | Path | None = None,
        id_map_path: str | Path | None = None,
    ) -> None:
        # Embedder initialization
        self.embedder: EmbedderPort = embedder or SentenceTransformerEmbedder()

        # Artifact paths
        self.index_path = Path(index_path or settings.index_path)
        self.id_map_path = Path(id_map_path or settings.id_map_path)

        if not self.index_path.is_file():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.id_map_path.is_file():
            raise FileNotFoundError(f"ID map file not found: {self.id_map_path}")

        # Load FAISS index
        self.index: faiss.Index = faiss.read_index(str(self.index_path))

        # Load ID map with backward compatibility
        with self.id_map_path.open("rb") as file:
            raw_id_map = pickle.load(file)

        # Allow list or dict for backward compatibility
        if isinstance(raw_id_map, dict):
            max_idx = max(raw_id_map.keys(), default=-1)
            self._id_map: List[int | None] = [None] * (max_idx + 1)
            for idx, doc_id in raw_id_map.items():
                self._id_map[idx] = doc_id
        elif isinstance(raw_id_map, Sequence):
            self._id_map = list(raw_id_map)  # shallow copy for safety
        else:
            raise TypeError(
                "id_map must be either a list or dict mapping FAISS indices to document IDs"
            )

        # Validation
        if self.index.d != self.embedder.DIM:
            raise ValueError(
                f"Dimension mismatch: FAISS index ({self.index.d}) vs embedder ({self.embedder.DIM})"
            )

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[int], List[float]]:
        """Retrieve the top-k most similar documents.

        Parameters
        ----------
        query : str
            The query string to search.
        k : int, default 5
            Number of top results to retrieve.

        Returns
        -------
        Tuple[List[int], List[float]]
            Parallel lists containing document IDs and similarity scores.

        Notes
        -----
        Returns empty lists if `k <= 0`.
        """
        if k <= 0:
            return [], []

        # FAISS expects shape (1, dimension)
        query_vec = np.asarray([self.embedder.embed(query)], dtype="float32")
        scores, indices = self.index.search(query_vec, min(k, self.index.ntotal))

        scores, indices = scores[0], indices[0]

        ids: List[int] = []
        sim_scores: List[float] = []

        for faiss_idx, score in zip(indices, scores):
            if faiss_idx == -1:
                continue  # FAISS fills with -1 if fewer than k results found
            doc_id = self._id_map[faiss_idx]
            if doc_id is None:
                continue  # Skip if ID mapping is missing

            ids.append(int(doc_id))
            sim_scores.append(float(score))

        return ids, sim_scores

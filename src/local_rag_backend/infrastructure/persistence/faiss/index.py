"""
RAG Prototype - Intrinsical-AI (c) 2025
Author: Pablo Pintor
License: MIT

Adapter for vector storage and search using FAISS.
"""

from __future__ import annotations

import pickle  # nosec B403
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


class FaissIndex:
    """Manages a FAISS index and its corresponding ID map."""

    def __init__(self, index_path: str | Path, id_map_path: str | Path, dim: int = 384):
        self.index_path = Path(index_path)
        self.id_map_path = Path(id_map_path)
        self.dim = dim
        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        """Load index and ID map from disk, or initialize if they don't exist."""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexFlatL2(self.dim)

        if self.id_map_path.exists():
            with self.id_map_path.open("rb") as f:
                self.id_map = pickle.load(f)  # nosec B301
        else:
            self.id_map = []

    def add_to_index(self, ids: list[int], embeddings: list[Sequence[float]]) -> None:
        """Add new vectors to the index and save."""
        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.shape[1] != self.index.d:
            raise ValueError(
                f"FAISS dim mismatch: vector dimension {vectors.shape[1]} != index dimension {self.index.d}"
            )

        self.index.add(vectors)
        self.id_map.extend(ids)
        self.save()

    def search(
        self, query_vector: Sequence[float], k: int
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Search the index for the k nearest neighbors."""
        query_np = np.asarray([query_vector], dtype="float32")
        distances, indices = self.index.search(query_np, k)
        return indices[0], distances[0]

    def save(self) -> None:
        """Save the index and ID map to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with self.id_map_path.open("wb") as f:
            pickle.dump(self.id_map, f)

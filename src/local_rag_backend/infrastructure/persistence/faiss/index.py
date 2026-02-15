# src/infrastructure/persistence/faiss/index.py
"""
Adapter for vector storage and search using FAISS.
"""

from __future__ import annotations

import pickle  # nosec B403
from pathlib import Path
from typing import TYPE_CHECKING

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
        self._faiss = None
        self._vectors: NDArray[np.float32] | None = None  # only used when FAISS isn't available
        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        """Load index and ID map from disk, or initialize if they don't exist."""
        try:
            import faiss

            self._faiss = faiss
        except ImportError:
            self._faiss = None

        if self._faiss is not None:
            if self.index_path.exists():
                self.index = self._faiss.read_index(str(self.index_path))
            else:
                self.index = self._faiss.IndexFlatL2(self.dim)
        else:
            # Pure-numpy fallback for environments without FAISS.
            if self.index_path.exists():
                try:
                    with self.index_path.open("rb") as f:
                        arr = np.load(f, allow_pickle=False)
                except Exception as e:  # pragma: no cover
                    raise RuntimeError(
                        "FAISS is not installed and the on-disk index is not readable via the "
                        "numpy fallback. Install the 'dense' extra (e.g. `uv sync --extra dense`)."
                    ) from e
                vectors = np.asarray(arr, dtype="float32")
                if vectors.ndim != 2 or vectors.shape[1] != self.dim:
                    raise ValueError(
                        f"Index dim mismatch: loaded vectors have shape {vectors.shape}, expected (*, {self.dim})"
                    )
                self._vectors = vectors
            else:
                self._vectors = np.empty((0, self.dim), dtype="float32")

        if self.id_map_path.exists():
            with self.id_map_path.open("rb") as f:
                self.id_map = pickle.load(f)  # nosec B301
        else:
            self.id_map = []

    def add_to_index(self, ids: list[int], embeddings: list[Sequence[float]]) -> None:
        """Add new vectors to the index and save."""
        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array-like (n, dim)")

        if self._faiss is not None:
            if vectors.shape[1] != self.index.d:
                raise ValueError(
                    f"FAISS dim mismatch: vector dimension {vectors.shape[1]} != index dimension {self.index.d}"
                )
            self.index.add(vectors)
        else:
            if vectors.shape[1] != self.dim:
                raise ValueError(
                    f"FAISS dim mismatch: vector dimension {vectors.shape[1]} != index dimension {self.dim}"
                )
            assert self._vectors is not None
            self._vectors = np.vstack([self._vectors, vectors]) if len(self._vectors) else vectors

        self.id_map.extend(ids)
        self.save()

    def search(
        self, query_vector: Sequence[float], k: int
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Search the index for the k nearest neighbors."""
        if k <= 0:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32)

        query_np = np.asarray(query_vector, dtype="float32").reshape(1, -1)

        if self._faiss is not None:
            distances, indices = self.index.search(query_np, k)
            return indices[0], distances[0]

        assert self._vectors is not None
        if self._vectors.size == 0:
            idxs = np.full((k,), -1, dtype=np.int64)
            dists = np.full((k,), np.inf, dtype=np.float32)
            return idxs, dists

        if query_np.shape[1] != self.dim:
            raise ValueError(
                f"Query dim mismatch: query dimension {query_np.shape[1]} != index dimension {self.dim}"
            )

        # Brute-force L2 distances
        diffs = self._vectors - query_np[0]
        dists_all = np.sum(diffs * diffs, axis=1).astype(np.float32)
        order = np.argsort(dists_all)[: min(k, len(dists_all))]

        idxs = np.full((k,), -1, dtype=np.int64)
        dists = np.full((k,), np.inf, dtype=np.float32)
        idxs[: len(order)] = order.astype(np.int64)
        dists[: len(order)] = dists_all[order]
        return idxs, dists

    def save(self) -> None:
        """Save the index and ID map to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if self._faiss is not None:
            self._faiss.write_index(self.index, str(self.index_path))
        else:
            assert self._vectors is not None
            with self.index_path.open("wb") as f:
                # Persist as .npy into the configured path (extension agnostic).
                np.save(f, self._vectors, allow_pickle=False)
        with self.id_map_path.open("wb") as f:
            pickle.dump(self.id_map, f)

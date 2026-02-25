from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


class FaissEngine:
    def __init__(self) -> None:
        try:
            import faiss
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "FAISS backend requested but faiss is not installed. "
                "Install the 'dense' extra (e.g. `uv sync --extra dense`)."
            ) from e

        self._faiss = faiss
        self._dim = 0
        self.index = None

    @property
    def backend(self) -> str:
        return "faiss"

    @property
    def dim(self) -> int:
        return int(self._dim)

    @property
    def ntotal(self) -> int:
        return int(getattr(self.index, "ntotal", 0) or 0)

    def infer_dim_or_raise(self, index_path: Path) -> int:
        if not index_path.exists():
            raise ValueError("dim is required when creating a new index (no existing index file).")
        idx = self._faiss.read_index(str(index_path))
        d = int(getattr(idx, "d", 0) or 0)
        if d <= 0:
            raise ValueError("Invalid FAISS index dim.")
        return d

    def load_or_initialize(self, index_path: Path, dim: int) -> None:
        self._dim = int(dim)
        if index_path.exists():
            self.index = self._faiss.read_index(str(index_path))
            if getattr(self.index, "d", None) != self._dim:
                raise ValueError(
                    "FAISS dim mismatch: "
                    f"loaded index dimension {getattr(self.index, 'd', None)} != expected {self._dim}"
                )
        else:
            self.index = self._faiss.IndexFlatL2(self._dim)

    def add(self, vectors: NDArray[np.float32]) -> None:
        if self.index is None:  # pragma: no cover
            raise RuntimeError("FAISS index unexpectedly uninitialized in add")
        if vectors.shape[1] != self.index.d:
            raise ValueError(
                f"FAISS dim mismatch: vector dimension {vectors.shape[1]} != index dimension {self.index.d}"
            )
        self.index.add(vectors)

    def delete_keep_positions(self, keep_positions: Sequence[int]) -> None:
        if self.index is None:  # pragma: no cover
            raise RuntimeError("FAISS index unexpectedly uninitialized in delete")
        if hasattr(self.index, "reconstruct_n"):
            all_vecs = self.index.reconstruct_n(0, self.index.ntotal)
        else:  # pragma: no cover
            all_vecs = np.vstack(
                [self.index.reconstruct(i) for i in range(self.index.ntotal)]
            ).astype("float32", copy=False)

        kept_vecs = np.asarray(all_vecs[keep_positions], dtype="float32")
        new_index = self._faiss.IndexFlatL2(self._dim)
        if len(kept_vecs):
            new_index.add(kept_vecs)
        self.index = new_index

    def rebuild(self, vectors: NDArray[np.float32]) -> None:
        new_index = self._faiss.IndexFlatL2(self._dim)
        if len(vectors):
            new_index.add(vectors)
        self.index = new_index

    def search(
        self, query_np: NDArray[np.float32], k: int
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        if self.index is None:  # pragma: no cover
            raise RuntimeError("FAISS index unexpectedly uninitialized in search")
        distances, indices = self.index.search(query_np, k)
        return indices[0], distances[0]

    def save(self, index_path: Path) -> None:
        if self.index is None:  # pragma: no cover
            raise RuntimeError("FAISS index unexpectedly uninitialized in save")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_index = index_path.with_name(index_path.name + ".tmp")
        self._faiss.write_index(self.index, str(tmp_index))
        os.replace(tmp_index, index_path)

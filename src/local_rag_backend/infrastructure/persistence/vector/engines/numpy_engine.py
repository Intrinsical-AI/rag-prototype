from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


class NumpyEngine:
    def __init__(self) -> None:
        self._dim = 0
        self._vectors: NDArray[np.float32] | None = None

    @property
    def backend(self) -> str:
        return "numpy"

    @property
    def dim(self) -> int:
        return int(self._dim)

    @property
    def ntotal(self) -> int:
        if self._vectors is None:
            return 0
        return int(self._vectors.shape[0])

    def infer_dim_or_raise(self, index_path: Path) -> int:
        if not index_path.exists():
            raise ValueError("dim is required when creating a new index (no existing index file).")

        try:
            with index_path.open("rb") as f:
                arr = np.load(f, allow_pickle=False)
        except Exception as e:
            raise RuntimeError("Unable to infer index dim from existing index file.") from e
        vectors = np.asarray(arr, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] <= 0:
            raise ValueError("Invalid on-disk vectors shape for dim inference.")
        return int(vectors.shape[1])

    def load_or_initialize(self, index_path: Path, dim: int) -> None:
        self._dim = int(dim)
        if index_path.exists():
            try:
                with index_path.open("rb") as f:
                    arr = np.load(f, allow_pickle=False)
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "FAISS is not installed and the on-disk index is not readable via the "
                    "numpy fallback. Install the 'dense' extra (e.g. `uv sync --extra dense`)."
                ) from e
            vectors = np.asarray(arr, dtype="float32")
            if vectors.ndim != 2 or vectors.shape[1] != self._dim:
                raise ValueError(
                    f"Index dim mismatch: loaded vectors have shape {vectors.shape}, expected (*, {self._dim})"
                )
            self._vectors = vectors
        else:
            self._vectors = np.empty((0, self._dim), dtype="float32")

    def add(self, vectors: NDArray[np.float32]) -> None:
        if vectors.shape[1] != self._dim:
            raise ValueError(
                f"FAISS dim mismatch: vector dimension {vectors.shape[1]} != index dimension {self._dim}"
            )
        if self._vectors is None:  # pragma: no cover
            raise RuntimeError("Numpy vectors unexpectedly uninitialized in add")
        self._vectors = np.vstack([self._vectors, vectors]) if len(self._vectors) else vectors

    def delete_keep_positions(self, keep_positions: Sequence[int]) -> None:
        if self._vectors is None:  # pragma: no cover
            raise RuntimeError("Numpy vectors unexpectedly uninitialized in delete")
        self._vectors = np.asarray(self._vectors[keep_positions], dtype="float32")

    def rebuild(self, vectors: NDArray[np.float32]) -> None:
        self._vectors = (
            np.asarray(vectors, dtype="float32")
            if len(vectors)
            else np.empty((0, self._dim), dtype="float32")
        )

    def search(
        self, query_np: NDArray[np.float32], k: int
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        if self._vectors is None:  # pragma: no cover
            raise RuntimeError("Numpy vectors unexpectedly uninitialized in search")
        if self._vectors.size == 0:
            idxs = np.full((k,), -1, dtype=np.int64)
            dists = np.full((k,), np.inf, dtype=np.float32)
            return idxs, dists

        if query_np.shape[1] != self._dim:
            raise ValueError(
                f"Query dim mismatch: query dimension {query_np.shape[1]} != index dimension {self._dim}"
            )

        diffs = self._vectors - query_np[0]
        dists_all = np.sum(diffs * diffs, axis=1).astype(np.float32)
        order = np.argsort(dists_all)[: min(k, len(dists_all))]

        idxs = np.full((k,), -1, dtype=np.int64)
        dists = np.full((k,), np.inf, dtype=np.float32)
        idxs[: len(order)] = order.astype(np.int64)
        dists[: len(order)] = dists_all[order]
        return idxs, dists

    def save(self, index_path: Path) -> None:
        if self._vectors is None:  # pragma: no cover
            raise RuntimeError("Numpy vectors unexpectedly uninitialized in save")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=index_path.name + ".",
            suffix=".tmp",
            dir=index_path.parent,
            delete=False,
        ) as tmp:
            np.save(tmp, self._vectors, allow_pickle=False)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, index_path)

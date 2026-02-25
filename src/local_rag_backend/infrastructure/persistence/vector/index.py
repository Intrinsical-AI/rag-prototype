"""
Vector index persistence for dense/hybrid retrieval.

Security + data-consistency notes:
- The ID map is persisted as JSON (list[int]) to avoid unsafe deserialization.
- Index + id-map writes are atomic.
- A cross-process lock protects concurrent writers and mitigates lost updates.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from local_rag_backend.core.services.file_lock import exclusive_file_lock
from local_rag_backend.infrastructure.persistence.shared.id_map_json import (
    load_id_map_json,
    save_id_map_json,
)
from local_rag_backend.infrastructure.persistence.vector.factory import build_vector_engine

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import NDArray


@contextmanager
def _exclusive_file_lock(lock_path: Path) -> Iterator[None]:
    with exclusive_file_lock(
        lock_path,
        error_message=(
            f"Unable to acquire vector file lock at {lock_path}. "
            "Refusing to mutate index state without a cross-process lock."
        ),
    ):
        yield


class VectorIndex:
    """Manages a vector index engine and its corresponding ID map."""

    def __init__(
        self,
        index_path: str | Path,
        id_map_path: str | Path,
        dim: int | None = 384,
        *,
        backend: str = "auto",
    ):
        self.index_path = Path(index_path)
        self.id_map_path = Path(id_map_path)
        self.dim = dim if dim is not None else 0

        self.engine = build_vector_engine(backend=backend)
        self._state_lock = threading.RLock()
        self._lock_path = self.index_path.with_name(self.index_path.name + ".lock")

        if dim is None:
            self.dim = self.engine.infer_dim_or_raise(self.index_path)
        self._load_or_initialize()

    @property
    def backend(self) -> str:
        return self.engine.backend

    @property
    def ntotal(self) -> int:
        with self._state_lock:
            return int(self.engine.ntotal)

    @contextmanager
    def _locked_write(self) -> Iterator[None]:
        with self._state_lock, _exclusive_file_lock(self._lock_path):
            yield

    def _load_or_initialize(self) -> None:
        with self._locked_write():
            self._load_or_initialize_locked()

    def _load_or_initialize_locked(self) -> None:
        self.engine.load_or_initialize(self.index_path, int(self.dim))
        self.id_map = load_id_map_json(self.id_map_path)

    def add_to_index(self, ids: list[int], embeddings: list[Sequence[float]]) -> None:
        if len(ids) != len(embeddings):
            raise ValueError(
                f"ids/embeddings length mismatch: {len(ids)} ids != {len(embeddings)} embeddings"
            )
        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array-like (n, dim)")

        with self._locked_write():
            self._load_or_initialize_locked()
            try:
                self.engine.add(vectors)
                self.id_map.extend(ids)
                self._save_locked()
            except Exception:
                with suppress(Exception):  # pragma: no cover
                    self._load_or_initialize_locked()
                raise

    def delete_ids(self, ids: Sequence[int]) -> int:
        to_delete = {int(x) for x in ids}
        if not to_delete:
            return 0

        with self._locked_write():
            self._load_or_initialize_locked()

            if not self.id_map:
                return 0

            keep_positions = [i for i, doc_id in enumerate(self.id_map) if doc_id not in to_delete]
            deleted = len(self.id_map) - len(keep_positions)
            if deleted <= 0:
                return 0

            self.engine.delete_keep_positions(keep_positions)
            self.id_map = [self.id_map[i] for i in keep_positions]
            self._save_locked()
            return deleted

    def rebuild(self, ids: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
        if len(ids) != len(vectors):
            raise ValueError(f"ids/vectors length mismatch: {len(ids)} != {len(vectors)}")
        vecs = np.asarray(list(vectors), dtype="float32")
        if vecs.ndim != 2 and len(ids):
            raise ValueError("Vectors must be a 2D array-like (n, dim)")
        if len(ids) and vecs.shape[1] != self.dim:
            raise ValueError(
                f"Dim mismatch: vectors have dim {vecs.shape[1]} but index dim is {self.dim}"
            )

        with self._locked_write():
            self.engine.load_or_initialize(self.index_path, int(self.dim))
            self.engine.rebuild(vecs)
            self.id_map = [int(x) for x in ids]
            self._save_locked()

    def search(
        self, query_vector: Sequence[float], k: int
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        if k <= 0:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32)

        query_np = np.asarray(query_vector, dtype="float32").reshape(1, -1)

        with self._state_lock:
            return self.engine.search(query_np, k)

    def save(self) -> None:
        with self._locked_write():
            self._save_locked()

    def _save_locked(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine.save(self.index_path)
        save_id_map_json(self.id_map_path, list(self.id_map))

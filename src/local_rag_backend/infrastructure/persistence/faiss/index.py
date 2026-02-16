# src/infrastructure/persistence/faiss/index.py
"""
Vector index persistence for dense/hybrid retrieval.

Security + data-consistency notes:
- The ID map is persisted as JSON (list[int]) to avoid unsafe deserialization.
- Index + id-map writes are atomic (temp file + os.replace).
- A cross-process lock (best-effort stdlib) protects against concurrent writers
  corrupting files and mitigates multi-worker "lost updates" by reloading the
  latest on-disk state under the lock before appending.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from local_rag_backend.core.services.file_lock import exclusive_file_lock

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import NDArray


def _looks_like_pickle(data: bytes) -> bool:
    # Pickle protocol v2+ starts with 0x80 <protocol>.
    return data.startswith(b"\x80")


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use a temp file in the same directory so `os.replace` is atomic.
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False
    ) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _atomic_write_text(path: Path, content: str) -> None:
    _atomic_write_bytes(path, content.encode("utf-8"))


@contextmanager
def _exclusive_file_lock(lock_path: Path) -> Iterator[None]:
    """
    Cross-process exclusive lock using only the stdlib.

    - POSIX: fcntl.flock
    - Windows: msvcrt.locking
    - Else: fail-closed
    """
    with exclusive_file_lock(
        lock_path,
        error_message=(
            f"Unable to acquire FAISS file lock at {lock_path}. "
            "Refusing to mutate index state without a cross-process lock."
        ),
    ):
        yield


class FaissIndex:
    """Manages a FAISS index (or numpy fallback) and its corresponding ID map."""

    def __init__(self, index_path: str | Path, id_map_path: str | Path, dim: int | None = 384):
        self.index_path = Path(index_path)
        self.id_map_path = Path(id_map_path)
        self.dim = dim if dim is not None else 0

        self._faiss = None
        self._vectors: NDArray[np.float32] | None = None  # numpy fallback when FAISS isn't present
        self._state_lock = threading.RLock()
        self._lock_path = self.index_path.with_name(self.index_path.name + ".lock")

        try:
            import faiss

            self._faiss = faiss
        except ImportError:
            self._faiss = None

        if dim is None:
            self.dim = self._infer_dim_or_raise()
        self._load_or_initialize()

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss is not None else "numpy"

    @property
    def ntotal(self) -> int:
        with self._state_lock:
            if self._faiss is not None:
                return int(getattr(self.index, "ntotal", 0) or 0)
            if self._vectors is None:
                return 0
            return int(self._vectors.shape[0])

    def _infer_dim_or_raise(self) -> int:
        if not self.index_path.exists():
            raise ValueError("dim is required when creating a new index (no existing index file).")

        # If FAISS is available and the file is a FAISS index, prefer it.
        if self._faiss is not None:
            idx = self._faiss.read_index(str(self.index_path))
            d = int(getattr(idx, "d", 0) or 0)
            if d <= 0:
                raise ValueError("Invalid FAISS index dim.")
            return d

        # Numpy fallback: load array header.
        try:
            with self.index_path.open("rb") as f:
                arr = np.load(f, allow_pickle=False)
        except Exception as e:
            raise RuntimeError("Unable to infer index dim from existing index file.") from e
        vectors = np.asarray(arr, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] <= 0:
            raise ValueError("Invalid on-disk vectors shape for dim inference.")
        return int(vectors.shape[1])

    def _load_or_initialize(self) -> None:
        with self._state_lock, _exclusive_file_lock(self._lock_path):
            self._load_or_initialize_locked()

    def _load_or_initialize_locked(self) -> None:
        """(Re)load on-disk state. Expects locks to be held."""
        if self._faiss is not None:
            if self.index_path.exists():
                self.index = self._faiss.read_index(str(self.index_path))
                if getattr(self.index, "d", None) != self.dim:
                    raise ValueError(
                        f"FAISS dim mismatch: loaded index dimension {getattr(self.index, 'd', None)} != expected {self.dim}"
                    )
            else:
                self.index = self._faiss.IndexFlatL2(self.dim)
        else:
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

        self.id_map = self._load_id_map_locked()

    def _load_id_map_locked(self) -> list[int]:
        """Load an ID-map from disk in a safe format (JSON list[int])."""
        if not self.id_map_path.exists():
            return []

        raw = self.id_map_path.read_bytes()
        if not raw:
            raise ValueError("Invalid id_map format: empty file.")
        if _looks_like_pickle(raw):
            raise RuntimeError(
                f"Unsafe pickle id_map detected at {self.id_map_path}. "
                "Delete it and rebuild the index (or migrate it to JSON)."
            )

        try:
            loaded = json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise ValueError(
                f"Invalid id_map format at {self.id_map_path}: expected JSON list[int]."
            ) from e

        if not isinstance(loaded, list) or not all(isinstance(x, int) for x in loaded):
            raise ValueError("Invalid id_map format: expected a JSON list[int].")
        return loaded

    def add_to_index(self, ids: list[int], embeddings: list[Sequence[float]]) -> None:
        """Add new vectors to the index and save."""
        if len(ids) != len(embeddings):
            raise ValueError(
                f"ids/embeddings length mismatch: {len(ids)} ids != {len(embeddings)} embeddings"
            )
        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array-like (n, dim)")

        with self._state_lock, _exclusive_file_lock(self._lock_path):
            # Reload under the lock so multi-worker ingestion appends to the latest state.
            self._load_or_initialize_locked()
            try:
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
                    self._vectors = (
                        np.vstack([self._vectors, vectors]) if len(self._vectors) else vectors
                    )

                self.id_map.extend(ids)
                self._save_locked()
            except Exception:
                # Critical: if persistence fails after mutating the in-memory index/id_map,
                # keep the process consistent by reloading the last known on-disk state.
                with suppress(Exception):  # pragma: no cover
                    self._load_or_initialize_locked()
                raise

    def delete_ids(self, ids: Sequence[int]) -> int:
        """
        Delete vectors for given document IDs.

        Notes:
        - This may rebuild the underlying index (O(n)).
        - Deletes all occurrences if an ID is present multiple times in the id_map.
        """
        to_delete = {int(x) for x in ids}
        if not to_delete:
            return 0

        with self._state_lock, _exclusive_file_lock(self._lock_path):
            self._load_or_initialize_locked()

            if not self.id_map:
                return 0

            keep_positions = [i for i, doc_id in enumerate(self.id_map) if doc_id not in to_delete]
            deleted = len(self.id_map) - len(keep_positions)
            if deleted <= 0:
                return 0

            # Rebuild vectors/index from the kept positions.
            if self._faiss is not None:
                # Reconstruct vectors from the flat index. IndexFlatL2 supports reconstruct.
                # Prefer reconstruct_n when available.
                if hasattr(self.index, "reconstruct_n"):
                    all_vecs = self.index.reconstruct_n(0, self.index.ntotal)
                else:  # pragma: no cover
                    all_vecs = np.vstack(
                        [self.index.reconstruct(i) for i in range(self.index.ntotal)]
                    ).astype("float32", copy=False)

                kept_vecs = np.asarray(all_vecs[keep_positions], dtype="float32")
                new_index = self._faiss.IndexFlatL2(self.dim)
                if len(kept_vecs):
                    new_index.add(kept_vecs)
                self.index = new_index
            else:
                assert self._vectors is not None
                self._vectors = np.asarray(self._vectors[keep_positions], dtype="float32")

            self.id_map = [self.id_map[i] for i in keep_positions]
            self._save_locked()
            return deleted

    def rebuild(self, ids: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
        """Rebuild the full index from scratch (idempotent)."""
        if len(ids) != len(vectors):
            raise ValueError(f"ids/vectors length mismatch: {len(ids)} != {len(vectors)}")
        vecs = np.asarray(list(vectors), dtype="float32")
        if vecs.ndim != 2 and len(ids):
            raise ValueError("Vectors must be a 2D array-like (n, dim)")
        if len(ids) and vecs.shape[1] != self.dim:
            raise ValueError(
                f"Dim mismatch: vectors have dim {vecs.shape[1]} but index dim is {self.dim}"
            )

        with self._state_lock, _exclusive_file_lock(self._lock_path):
            if self._faiss is not None:
                self.index = self._faiss.IndexFlatL2(self.dim)
                if len(ids):
                    self.index.add(vecs)
            else:
                self._vectors = (
                    np.asarray(vecs, dtype="float32")
                    if len(ids)
                    else np.empty((0, self.dim), dtype="float32")
                )

            self.id_map = [int(x) for x in ids]
            self._save_locked()

    def search(
        self, query_vector: Sequence[float], k: int
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Search the index for the k nearest neighbors."""
        if k <= 0:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32)

        query_np = np.asarray(query_vector, dtype="float32").reshape(1, -1)

        with self._state_lock:
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
        with self._state_lock, _exclusive_file_lock(self._lock_path):
            self._save_locked()

    def _save_locked(self) -> None:
        """Same as `save`, but expects locks to already be held."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        if self._faiss is not None:
            tmp_index = self.index_path.with_name(self.index_path.name + ".tmp")
            self._faiss.write_index(self.index, str(tmp_index))
            os.replace(tmp_index, self.index_path)
        else:
            assert self._vectors is not None
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=self.index_path.name + ".",
                suffix=".tmp",
                dir=self.index_path.parent,
                delete=False,
            ) as tmp:
                np.save(tmp, self._vectors, allow_pickle=False)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, self.index_path)

        _atomic_write_text(self.id_map_path, json.dumps(list(self.id_map)))

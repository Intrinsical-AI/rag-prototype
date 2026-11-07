"""Read-only vector search over already verified snapshot bytes."""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from typing import Any

import numpy as np

from local_rag_backend.core.domain.types import DocId
from local_rag_backend.infrastructure.persistence.read_snapshot import (
    SnapshotError,
    VerifiedSnapshot,
)
from local_rag_backend.infrastructure.persistence.vector.manifest import validate_manifest
from local_rag_backend.infrastructure.retrieval.scoring import normalize_min_max_scores


class SnapshotVectorStorage:
    """No constructor, search or refusal path opens a live index or lock."""

    def __init__(self, snapshot: VerifiedSnapshot) -> None:
        if (
            snapshot.vector_bytes is None
            or snapshot.ids_bytes is None
            or snapshot.native_manifest is None
        ):
            raise SnapshotError("Snapshot vector artifacts are missing")
        raw_ids = json.loads(snapshot.ids_bytes)
        if (
            not isinstance(raw_ids, list)
            or any(not isinstance(item, str) or not item.strip() for item in raw_ids)
            or len(set(raw_ids)) != len(raw_ids)
        ):
            raise SnapshotError("Snapshot vector IDs are invalid")
        self._ids = tuple(DocId(item) for item in raw_ids)
        if set(self._ids) != {doc.id for doc in snapshot.documents}:
            raise SnapshotError("Snapshot SQL/vector identity drift")
        self._index: Any = None
        self._vectors: Any = None
        backend = snapshot.manifest.vector_backend
        if backend == "numpy":
            try:
                vectors = np.load(io.BytesIO(snapshot.vector_bytes), allow_pickle=False)
            except (ValueError, EOFError, OSError) as exc:
                raise SnapshotError("Snapshot numpy vectors cannot be read") from exc
            if (
                not isinstance(vectors, np.ndarray)
                or vectors.ndim != 2
                or vectors.shape[1] < 1
                or not np.isfinite(vectors).all()
            ):
                raise SnapshotError("Snapshot numpy vector shape or values are invalid")
            self._vectors = np.asarray(vectors, dtype=np.float32)
            self.dim = int(vectors.shape[1])
            count = len(vectors)
        elif backend == "faiss":
            try:
                import faiss
            except ImportError as exc:
                raise SnapshotError("Snapshot requires the dense extra for FAISS") from exc
            try:
                self._index = faiss.deserialize_index(
                    np.frombuffer(snapshot.vector_bytes, dtype=np.uint8).copy()
                )
            except RuntimeError as exc:
                raise SnapshotError("Snapshot FAISS vectors cannot be read") from exc
            self.dim = int(self._index.d)
            count = int(self._index.ntotal)
        else:
            raise SnapshotError("Snapshot vector backend is invalid")
        mismatches, errors = validate_manifest(
            manifest=snapshot.native_manifest,
            expected_config=snapshot.manifest.embedding_config,
            actual_dimension=self.dim,
            actual_index_backend=str(backend),
        )
        if count != len(self._ids) or mismatches or errors:
            raise SnapshotError("Snapshot vector manifest or count drift")

    @property
    def ntotal(self) -> int:
        return len(self._ids)

    def similar(self, vector: Sequence[float], k: int) -> list[tuple[DocId, float]]:
        query = np.asarray(vector, dtype=np.float32)
        if query.shape != (self.dim,) or not np.isfinite(query).all():
            raise SnapshotError("Query embedding dimension or values are invalid")
        if self.ntotal == 0:
            return []
        if self._index is not None:
            distances, indices = self._index.search(query.reshape(1, -1), min(k, self.ntotal))
            positions = indices[0].tolist()
            distance_values = distances[0].tolist()
        else:
            distances = np.sum((self._vectors - query) ** 2, axis=1)
            positions = np.argsort(distances)[:k].tolist()
            distance_values = [float(distances[index]) for index in positions]
        valid = [
            (index, float(distance))
            for index, distance in zip(positions, distance_values, strict=True)
            if 0 <= index < self.ntotal
        ]
        scores = normalize_min_max_scores(
            [1.0 / (1.0 + distance) for _, distance in valid], flat_value=0.0, singleton_value=1.0
        )
        return [(self._ids[index], score) for (index, _), score in zip(valid, scores, strict=True)]

    def upsert(self, ids: Sequence[DocId], vectors: Sequence[Sequence[float]]) -> None:
        del ids, vectors
        raise SnapshotError("Frozen vector storage is read-only")

    def apply_delta_atomic(
        self, *, delete_ids: Sequence[DocId], upserts: Sequence[tuple[DocId, Sequence[float]]]
    ) -> None:
        del delete_ids, upserts
        raise SnapshotError("Frozen vector storage is read-only")

    def delete(self, ids: Sequence[DocId]) -> int:
        del ids
        raise SnapshotError("Frozen vector storage is read-only")

    def rebuild(self, ids: Sequence[DocId], vectors: Sequence[Sequence[float]]) -> None:
        del ids, vectors
        raise SnapshotError("Frozen vector storage is read-only")

"""Adapter for vector storage and search."""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_backend.core.domain.types import DocId
from local_rag_backend.core.ports import VectorRepoPort
from local_rag_backend.infrastructure.persistence.vector.index import VectorIndex
from local_rag_backend.infrastructure.persistence.vector.manifest import (
    create_manifest_if_missing_for_settings,
    expected_manifest_config_from_settings,
    manifest_path_for,
    overwrite_manifest_for_settings,
    read_manifest,
    validate_manifest,
)
from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray


class VectorStorage(VectorRepoPort):
    """Adapter for vector storage and search."""

    def __init__(
        self,
        index_path: str,
        id_map_path: str,
        dim: int | None = 384,
        *,
        backend: str | None = None,
    ):
        resolved_backend = str(backend or settings.vector_backend)
        self.vector_index = VectorIndex(
            index_path,
            id_map_path,
            dim,
            backend=resolved_backend,
        )

    def _expected_manifest_config(self) -> dict[str, str]:
        return expected_manifest_config_from_settings(settings)

    def _ensure_manifest(self, *, overwrite: bool) -> None:
        idx_path = getattr(self.vector_index, "index_path", None)
        dim_attr = getattr(self.vector_index, "dim", None)
        backend_attr = getattr(self.vector_index, "backend", None)
        if idx_path is None or dim_attr is None or backend_attr is None:
            return

        expected = self._expected_manifest_config()
        dim = int(dim_attr)
        backend = str(backend_attr)

        if overwrite:
            overwrite_manifest_for_settings(
                index_path=idx_path,
                expected=expected,
                dimension=dim,
                index_backend=backend,
            )
            return

        manifest_path = manifest_path_for(idx_path)
        manifest = read_manifest(manifest_path)
        if manifest is None:
            vectors = int(getattr(self.vector_index, "ntotal", 0) or 0)
            id_map = getattr(self.vector_index, "id_map", [])
            id_map_len = len(id_map) if isinstance(id_map, list) else 0
            if vectors > 0 or id_map_len > 0:
                raise RuntimeError(
                    "Index manifest missing for a non-empty index; rebuild is required "
                    "(hint: run `rag-rebuild-index` or POST /api/index/rebuild)."
                )

        create_manifest_if_missing_for_settings(
            index_path=idx_path,
            expected=expected,
            dimension=dim,
            index_backend=backend,
        )

        manifest = read_manifest(manifest_path)
        if manifest is None:  # pragma: no cover
            raise RuntimeError("Index manifest is missing after creation attempt.")
        mismatches, errors = validate_manifest(
            manifest=manifest,
            expected_config=expected,
            actual_dimension=dim,
            actual_index_backend=backend,
        )
        if errors or mismatches:
            raise RuntimeError(
                "Index manifest drift detected; rebuild is required "
                "(hint: run `rag-rebuild-index` or POST /api/index/rebuild)."
            )

    def upsert(self, ids: Sequence[DocId], vectors: Sequence[Sequence[float]]) -> None:
        self._ensure_manifest(overwrite=False)
        self.vector_index.add_to_index(list(ids), list(vectors))

    def apply_delta_atomic(
        self,
        *,
        delete_ids: Sequence[DocId],
        upserts: Sequence[tuple[DocId, Sequence[float]]],
    ) -> None:
        self._ensure_manifest(overwrite=False)
        self.vector_index.apply_delta_atomic(delete_ids=delete_ids, upserts=upserts)

    def delete(self, ids: Sequence[DocId]) -> int:
        self._ensure_manifest(overwrite=False)
        return self.vector_index.delete_ids(list(ids))

    def rebuild(self, ids: Sequence[DocId], vectors: Sequence[Sequence[float]]) -> None:
        self.vector_index.rebuild(ids, vectors)
        self._ensure_manifest(overwrite=True)

    def search(
        self, query_vector: Sequence[float], k: int
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        return self.vector_index.search(query_vector, k)

    def similar(self, vector: Sequence[float], k: int) -> list[tuple[DocId, float]]:
        indices, distances = self.search(vector, k)

        id_map = self.vector_index.id_map
        valid_results = [
            (id_map[i], float(d))
            for i, d in zip(indices, distances, strict=False)
            if i != -1 and 0 <= int(i) < len(id_map)
        ]
        if not valid_results:
            return []

        doc_ids, valid_distances = zip(*valid_results, strict=False)
        sim_raw = [1.0 / (1.0 + d) for d in valid_distances]
        min_s, max_s = min(sim_raw), max(sim_raw)

        if max_s == min_s:
            normalized_sims = [1.0] * len(sim_raw)
        else:
            normalized_sims = [(s - min_s) / (max_s - min_s) for s in sim_raw]

        return list(zip(doc_ids, normalized_sims, strict=False))

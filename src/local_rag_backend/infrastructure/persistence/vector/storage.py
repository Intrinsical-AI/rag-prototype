"""Adapter for vector storage and search."""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_backend.core.domain.embeddings import EmbeddingIdentity
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
from local_rag_backend.infrastructure.retrieval.scoring import normalize_min_max_scores

if TYPE_CHECKING:
    from collections.abc import Sequence

    from local_rag_backend.settings import Settings


class VectorStorage(VectorRepoPort):
    """Adapter for vector storage and search."""

    def __init__(
        self,
        index_path: str,
        id_map_path: str,
        dim: int | None = 384,
        *,
        backend: str | None = None,
        settings_obj: Settings,
        embedding_identity: EmbeddingIdentity | None = None,
    ):
        self._settings = settings_obj
        self._embedding_identity = embedding_identity
        if embedding_identity is not None:
            if dim is not None and dim != embedding_identity.dimension:
                raise ValueError("Embedding identity dimension does not match vector dimension")
            dim = embedding_identity.dimension
        resolved_backend = str(backend or self._settings.vector_backend)
        self.vector_index = VectorIndex(
            index_path,
            id_map_path,
            dim,
            backend=resolved_backend,
        )

    def _expected_manifest_config(self) -> dict[str, object]:
        return expected_manifest_config_from_settings(
            self._settings, embedding_identity=self._embedding_identity
        )

    def _ensure_manifest(self, *, overwrite: bool, create: bool = True) -> None:
        idx_path = self.vector_index.index_path
        expected = self._expected_manifest_config()
        dim = self.vector_index.dim
        backend = self.vector_index.backend

        if overwrite:
            overwrite_manifest_for_settings(
                index_path=idx_path,
                expected=expected,
                dimension=dim,
                index_backend=backend,
            )
            return

        manifest_path = manifest_path_for(idx_path)
        try:
            manifest = read_manifest(manifest_path)
        except (OSError, ValueError) as exc:
            raise RuntimeError("Index manifest invalid; rebuild is required.") from exc
        if manifest is None:
            if self.vector_index.ntotal > 0 or self.vector_index.id_map:
                raise RuntimeError(
                    "Index manifest missing for a non-empty index; rebuild is required "
                    "(hint: run `rag-rebuild-index` or POST /api/index/rebuild)."
                )

            if not create:
                return
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

    @property
    def ntotal(self) -> int:
        return self.vector_index.ntotal

    def upsert(self, ids: Sequence[DocId], vectors: Sequence[Sequence[float]]) -> None:
        self._ensure_manifest(overwrite=False, create=bool(ids))
        if not ids and not vectors:
            return
        self.vector_index.add_to_index(list(ids), list(vectors))

    def apply_delta_atomic(
        self,
        *,
        delete_ids: Sequence[DocId],
        upserts: Sequence[tuple[DocId, Sequence[float]]],
    ) -> None:
        self._ensure_manifest(overwrite=False, create=bool(upserts))
        self.vector_index.apply_delta_atomic(delete_ids=delete_ids, upserts=upserts)

    def delete(self, ids: Sequence[DocId]) -> int:
        self._ensure_manifest(overwrite=False, create=False)
        return self.vector_index.delete_ids(list(ids))

    def rebuild(self, ids: Sequence[DocId], vectors: Sequence[Sequence[float]]) -> None:
        self.vector_index.rebuild(ids, vectors)
        self._ensure_manifest(overwrite=True)

    def rebuild_from_batches(
        self,
        batches: Sequence[tuple[Sequence[DocId], Sequence[Sequence[float]]]],
    ) -> None:
        self.vector_index.rebuild_from_batches(batches)
        self._ensure_manifest(overwrite=True)

    def similar(self, vector: Sequence[float], k: int) -> list[tuple[DocId, float]]:
        self._ensure_manifest(overwrite=False, create=False)
        indices, distances, id_map = self.vector_index.search_with_snapshot(vector, k)
        valid_results = [
            (id_map[i], float(d))
            for i, d in zip(indices, distances, strict=False)
            if i != -1 and 0 <= int(i) < len(id_map)
        ]
        if not valid_results:
            return []

        doc_ids, valid_distances = zip(*valid_results, strict=False)
        sim_raw = [1.0 / (1.0 + d) for d in valid_distances]
        normalized_sims = normalize_min_max_scores(
            sim_raw,
            flat_value=0.0,
            singleton_value=1.0,
        )

        return list(zip(doc_ids, normalized_sims, strict=False))

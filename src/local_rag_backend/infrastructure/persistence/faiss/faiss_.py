# src/infrastructure/persistence/faiss/faiss_.py
"""
Adapter for vector storage and search using FAISS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_backend.core.ports import VectorRepoPort
from local_rag_backend.infrastructure.persistence.faiss.index import FaissIndex
from local_rag_backend.infrastructure.persistence.faiss.manifest import (
    build_expected_manifest_config,
    create_manifest_if_missing_for_settings,
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


class FaissVectorStorage(VectorRepoPort):
    """Adapter for vector storage and search using FAISS."""

    def __init__(self, index_path: str, id_map_path: str, dim: int | None = 384):
        self.faiss_index = FaissIndex(index_path, id_map_path, dim)

    def _expected_manifest_config(self) -> dict[str, str]:
        embedding_backend = "openai" if bool(settings.openai_api_key) else "sentence_transformers"
        embedding_model = (
            settings.openai_embedding_model
            if bool(settings.openai_api_key)
            else settings.st_embedding_model
        )
        return build_expected_manifest_config(
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            chunker_strategy=settings.ingest_chunk_strategy,
            chunker_version=settings.ingest_chunker_version,
        )

    def _ensure_manifest(self, *, overwrite: bool) -> None:
        """
        Keep drift detection meaningful:
        - For incremental writes (upsert/delete): create manifest if missing; refuse to overwrite drift.
        - For full rebuilds: overwrite to the new config.
        """
        # Some tests monkeypatch `FaissIndex` with a lightweight dummy object. In that case,
        # skip manifest persistence; production uses the real `FaissIndex`.
        idx_path = getattr(self.faiss_index, "index_path", None)
        dim_attr = getattr(self.faiss_index, "dim", None)
        backend_attr = getattr(self.faiss_index, "backend", None)
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
            vectors = int(getattr(self.faiss_index, "ntotal", 0) or 0)
            id_map = getattr(self.faiss_index, "id_map", [])
            id_map_len = len(id_map) if isinstance(id_map, list) else 0
            # Fail closed for legacy non-empty indexes without manifest.
            # Auto-creating here would "bless" unknown historical config and hide drift.
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

    def upsert(self, ids: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
        """Add vectors to the FAISS index."""
        # Guard first: if manifest drifts from current settings, fail before mutating index files.
        self._ensure_manifest(overwrite=False)
        self.faiss_index.add_to_index(list(ids), list(vectors))

    def delete(self, ids: Sequence[int]) -> None:
        """Delete vectors from the index (may rebuild the underlying index)."""
        # Guard first: if manifest drifts from current settings, fail before mutating index files.
        self._ensure_manifest(overwrite=False)
        self.faiss_index.delete_ids(list(ids))

    def rebuild(self, ids: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
        """Rebuild the full index from scratch (idempotent)."""
        self.faiss_index.rebuild(ids, vectors)
        self._ensure_manifest(overwrite=True)

    def search(
        self, query_vector: Sequence[float], k: int
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Perform a raw search in the FAISS index."""
        return self.faiss_index.search(query_vector, k)

    def similar(self, vector: Sequence[float], k: int) -> list[tuple[int, float]]:
        """Find similar items and return their IDs and normalized similarity scores."""
        indices, distances = self.search(vector, k)

        # Filter out invalid indices (-1) and guard against id_map/index mismatches.
        # A mismatch can happen if files are manually edited/corrupted; don't crash retrieval.
        id_map = self.faiss_index.id_map
        valid_results = [
            (id_map[i], float(d))
            for i, d in zip(indices, distances, strict=False)
            if i != -1 and 0 <= int(i) < len(id_map)
        ]
        if not valid_results:
            return []

        # Normalize distances to similarity scores [0, 1]
        doc_ids, valid_distances = zip(*valid_results, strict=False)
        sim_raw = [1.0 / (1.0 + d) for d in valid_distances]
        min_s, max_s = min(sim_raw), max(sim_raw)

        if max_s == min_s:
            normalized_sims = [1.0] * len(sim_raw)
        else:
            normalized_sims = [(s - min_s) / (max_s - min_s) for s in sim_raw]

        return list(zip(doc_ids, normalized_sims, strict=False))

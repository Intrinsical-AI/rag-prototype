"""Stable, PEP 561-typed embedding API for installed consumers.

Only names listed in ``__all__`` are part of the supported public boundary.
"""

from __future__ import annotations

from pathlib import Path

from local_rag_backend.core.errors import EmbeddingsBackendUnavailableError
from local_rag_backend.integrations.embeddings._contracts import (
    DEFAULT_EMBEDDING_LIMITS,
    EmbeddingLimits,
    EmbeddingService,
    EmbeddingStatus,
)


def create_embedding_service(
    config_path: str | Path | None = None,
    *,
    limits: EmbeddingLimits | None = None,
) -> EmbeddingService:
    """Create a reusable service; explicit config takes priority over RAG_CONFIG_PATH."""
    from local_rag_backend.integrations.embeddings._service import (
        create_embedding_service as _create_embedding_service,
    )

    return _create_embedding_service(config_path=config_path, limits=limits)


__all__ = [
    "DEFAULT_EMBEDDING_LIMITS",
    "EmbeddingLimits",
    "EmbeddingService",
    "EmbeddingStatus",
    "EmbeddingsBackendUnavailableError",
    "create_embedding_service",
]

"""Stable, PEP 561-typed embedding API for installed consumers.

Only names listed in ``__all__`` are part of the supported public boundary.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from threading import Lock

from local_rag_backend.core.errors import EmbeddingsBackendUnavailableError
from local_rag_backend.integrations.embeddings._contracts import (
    DEFAULT_EMBEDDING_LIMITS,
    EmbeddingLimits,
    EmbeddingService,
    EmbeddingStatus,
)

_CONFIG_ENV_VAR = "RAG_CONFIG_PATH"
_SERVICE_IMPORT_LOCK = Lock()


def create_embedding_service(
    config_path: str | Path | None = None,
    *,
    limits: EmbeddingLimits | None = None,
) -> EmbeddingService:
    """Create a reusable embedding service.

    ``config_path`` has priority over ``RAG_CONFIG_PATH``. The temporary environment
    assignment permits the eager settings module to initialize when this is the first
    RAG import in an installed process; provider construction receives the explicitly
    loaded settings object directly.
    """
    with _SERVICE_IMPORT_LOCK:
        settings_need_initialization = "local_rag_backend.settings" not in sys.modules
        temporary_config_path = (
            str(Path(config_path).expanduser())
            if config_path is not None and settings_need_initialization
            else None
        )
        had_previous = _CONFIG_ENV_VAR in os.environ
        previous = os.environ.get(_CONFIG_ENV_VAR)
        if temporary_config_path is not None:
            os.environ[_CONFIG_ENV_VAR] = temporary_config_path
        try:
            from local_rag_backend.integrations.embeddings._service import (
                create_embedding_service as _create_embedding_service,
            )
        finally:
            if temporary_config_path is not None:
                if had_previous and previous is not None:
                    os.environ[_CONFIG_ENV_VAR] = previous
                else:
                    os.environ.pop(_CONFIG_ENV_VAR, None)
    return _create_embedding_service(config_path=config_path, limits=limits)


__all__ = [
    "DEFAULT_EMBEDDING_LIMITS",
    "EmbeddingLimits",
    "EmbeddingService",
    "EmbeddingStatus",
    "EmbeddingsBackendUnavailableError",
    "create_embedding_service",
]

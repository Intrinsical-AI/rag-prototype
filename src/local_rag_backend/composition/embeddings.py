"""Composition helpers for dense embedder selection."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from local_rag_backend.core.errors import EmbeddingsBackendUnavailableError
from local_rag_backend.core.ports import EmbedderPort
from local_rag_backend.infrastructure.embeddings.cached import (
    ContentAddressedCachingEmbedder,
    resolve_embedding_cache_db_path,
)

if TYPE_CHECKING:
    from local_rag_backend.settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_DENSE_BACKEND_MESSAGE = (
    "Dense/hybrid retrieval requires an embeddings backend. "
    "Configure openai_api_key in config.yaml to use OpenAI embeddings, or install the "
    "'dense-st' extra for SentenceTransformers (e.g. `uv sync --extra dense-st`)."
)


def _settings_cfg_version(settings_obj: Settings) -> str:
    return str(getattr(settings_obj, "storage_profile", "") or "default")


def _build_default_openai_embedder() -> EmbedderPort:
    from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder

    return OpenAIEmbedder()


def _build_default_st_embedder(model_name: str) -> EmbedderPort:
    from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
        SentenceTransformerEmbedder,
    )

    return SentenceTransformerEmbedder(model_name=model_name)


def build_dense_embedder_from_settings(
    *,
    settings_obj: Settings,
    openai_embedder_factory: Callable[[], EmbedderPort] | None = None,
    st_embedder_factory: Callable[[str], EmbedderPort] | None = None,
    missing_backend_message: str | None = None,
) -> EmbedderPort:
    """Build the dense embedder stack selected by settings."""
    resolved_openai_factory = openai_embedder_factory or _build_default_openai_embedder
    resolved_st_factory = st_embedder_factory or _build_default_st_embedder
    backend_message = missing_backend_message or DEFAULT_DENSE_BACKEND_MESSAGE
    cache_db_path = resolve_embedding_cache_db_path(
        data_dir=Path(getattr(settings_obj, "data_dir", "data")),
        configured_path=getattr(settings_obj, "embedding_cache_db_path", None),
    )
    if settings_obj.openai_api_key:
        base = resolved_openai_factory()
        provider = "openai"
    else:
        try:
            base = resolved_st_factory(str(settings_obj.st_embedding_model))
        except RuntimeError as e:
            raise EmbeddingsBackendUnavailableError(backend_message) from e
        provider = "sentence_transformers"
    logger.info(
        "dense_embedder_selected provider=%s mode=%s cache_key=%s backend=%s cfg_version=%s",
        provider,
        str(getattr(settings_obj, "retrieval_mode", "unknown")),
        str(cache_db_path),
        str(getattr(settings_obj, "search_backend", "local_split")),
        _settings_cfg_version(settings_obj),
    )
    return ContentAddressedCachingEmbedder(
        base=base,
        cache_db_path=cache_db_path,
        disabled=bool(getattr(settings_obj, "disable_embedding_cache", False)),
    )

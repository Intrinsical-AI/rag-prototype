"""Configured implementation of the public embedding service contract."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from local_rag_backend.core.ports import EmbedderPort
from local_rag_backend.infrastructure.embeddings.cached import resolve_embedding_cache_db_path
from local_rag_backend.integrations.embeddings._contracts import (
    DEFAULT_EMBEDDING_LIMITS,
    EmbeddingLimits,
    EmbeddingProvider,
    EmbeddingService,
    EmbeddingStatus,
)
from local_rag_backend.integrations.embeddings._factory import (
    build_dense_embedder_from_settings,
)
from local_rag_backend.settings import Settings, load_settings_from_yaml


class _ConfiguredEmbeddingService:
    def __init__(
        self,
        *,
        embedder: EmbedderPort,
        settings_obj: Settings,
        limits: EmbeddingLimits,
    ) -> None:
        self._embedder = embedder
        self._limits = limits
        provider: EmbeddingProvider = (
            "openai" if settings_obj.openai_api_key else "sentence_transformers"
        )
        model = (
            str(settings_obj.openai_embedding_model)
            if provider == "openai"
            else str(settings_obj.st_embedding_model)
        )
        cache_db_path = resolve_embedding_cache_db_path(
            data_dir=Path(settings_obj.data_dir),
            configured_path=settings_obj.embedding_cache_db_path,
        )
        self._status: EmbeddingStatus = {
            "provider": provider,
            "model": model,
            "model_key": str(getattr(embedder, "model_key", f"{provider}:{model}:{embedder.dim}")),
            "dimension": int(embedder.dim),
            "synthetic": bool(
                provider == "sentence_transformers" and settings_obj.synthetic_embeddings
            ),
            "cache_enabled": not bool(settings_obj.disable_embedding_cache),
            "cache_db_path": str(cache_db_path),
            "limits": {
                "max_batch_size": limits.max_batch_size,
                "max_text_chars": limits.max_text_chars,
                "max_total_chars": limits.max_total_chars,
            },
        }

    def status(self) -> EmbeddingStatus:
        limits = self._status["limits"]
        return {
            "provider": self._status["provider"],
            "model": self._status["model"],
            "model_key": self._status["model_key"],
            "dimension": self._status["dimension"],
            "synthetic": self._status["synthetic"],
            "cache_enabled": self._status["cache_enabled"],
            "cache_db_path": self._status["cache_db_path"],
            "limits": {
                "max_batch_size": limits["max_batch_size"],
                "max_text_chars": limits["max_text_chars"],
                "max_total_chars": limits["max_total_chars"],
            },
        }

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if isinstance(texts, str | bytes):
            raise ValueError("texts must be a sequence of strings, not a string")
        batch = list(texts)
        if len(batch) > self._limits.max_batch_size:
            raise ValueError(
                f"embedding batch exceeds max_batch_size={self._limits.max_batch_size}"
            )

        total_chars = 0
        for index, text in enumerate(batch):
            if not isinstance(text, str):
                raise ValueError(f"texts[{index}] must be a string")
            if not text.strip():
                raise ValueError(f"texts[{index}] must not be blank")
            if len(text) > self._limits.max_text_chars:
                raise ValueError(
                    f"texts[{index}] exceeds max_text_chars={self._limits.max_text_chars}"
                )
            total_chars += len(text)
        if total_chars > self._limits.max_total_chars:
            raise ValueError(
                f"embedding batch exceeds max_total_chars={self._limits.max_total_chars}"
            )
        if not batch:
            return []

        raw_vectors = list(self._embedder.embed(batch))
        if len(raw_vectors) != len(batch):
            raise RuntimeError(
                f"Embedder returned {len(raw_vectors)} vectors for {len(batch)} texts."
            )

        vectors: list[list[float]] = []
        for index, raw_vector in enumerate(raw_vectors):
            try:
                vector = [float(value) for value in raw_vector]
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Embedder returned a non-numeric vector at index {index}."
                ) from exc
            if len(vector) != self._status["dimension"]:
                raise RuntimeError(
                    f"Embedder returned dimension {len(vector)} at index {index}; "
                    f"expected {self._status['dimension']}."
                )
            if not all(math.isfinite(value) for value in vector):
                raise RuntimeError(f"Embedder returned a non-finite vector at index {index}.")
            vectors.append(vector)
        return vectors


def create_embedding_service(
    config_path: str | Path | None = None,
    *,
    limits: EmbeddingLimits | None = None,
) -> EmbeddingService:
    """Create one reusable provider/cache stack from explicit or environment config."""
    settings_obj = load_settings_from_yaml(config_path)
    embedder = build_dense_embedder_from_settings(settings_obj=settings_obj)
    return cast(
        "EmbeddingService",
        _ConfiguredEmbeddingService(
            embedder=embedder,
            settings_obj=settings_obj,
            limits=limits or DEFAULT_EMBEDDING_LIMITS,
        ),
    )

"""Public typed contracts for the embedding integration API."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict, runtime_checkable

EmbeddingProvider = Literal["openai", "sentence_transformers"]


class EmbeddingLimitStatus(TypedDict):
    """JSON-safe limits reported by an embedding service."""

    max_batch_size: int
    max_text_chars: int
    max_total_chars: int


class EmbeddingStatus(TypedDict):
    """JSON-safe description of the configured embedding stack."""

    provider: EmbeddingProvider
    model: str
    model_key: str
    dimension: int
    synthetic: bool
    cache_enabled: bool
    cache_db_path: str
    limits: EmbeddingLimitStatus


@dataclass(frozen=True, slots=True)
class EmbeddingLimits:
    """Per-call resource limits enforced before invoking an embedding backend."""

    max_batch_size: int = 128
    max_text_chars: int = 32_768
    max_total_chars: int = 262_144

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if self.max_text_chars < 1:
            raise ValueError("max_text_chars must be at least 1")
        if self.max_total_chars < self.max_text_chars:
            raise ValueError("max_total_chars must be at least max_text_chars")


DEFAULT_EMBEDDING_LIMITS = EmbeddingLimits()


@runtime_checkable
class EmbeddingService(Protocol):
    """Stable installed-consumer interface for status and bounded embedding."""

    def status(self) -> EmbeddingStatus:
        """Return a JSON-serializable snapshot without secrets."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Validate and embed one bounded batch while preserving input order."""
        ...

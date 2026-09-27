"""Identity of one embedding space, shared by providers and persisted artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal

EmbeddingProvider = Literal["openai", "sentence_transformers"]
EMBEDDING_IMPLEMENTATION_VERSION = "1"


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    provider: EmbeddingProvider
    model: str
    dimension: int
    synthetic: bool = False
    implementation_version: str = EMBEDDING_IMPLEMENTATION_VERSION

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("Embedding dimension must be positive")
        if not self.model.strip() or not self.implementation_version.strip():
            raise ValueError("Embedding model and implementation version must not be blank")

    @property
    def model_key(self) -> str:
        """Unambiguous, deterministic key for content-addressed embedding caches."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def manifest_fields(self) -> dict[str, str | int | bool]:
        return {
            "embedding_backend": self.provider,
            "embedding_model": self.model,
            "dimension": self.dimension,
            "synthetic": self.synthetic,
            "implementation_version": self.implementation_version,
        }

# src/infrastructure/embeddings/openai.py
"""
OpenAI embeddings implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from openai import OpenAI

from local_rag_backend.core.ports import EmbedderPort
from local_rag_backend.infrastructure.llms.openai_chat import create_openai_client
from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from local_rag_backend.settings import Settings

_MODEL_DIM: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

Embedding = Sequence[float]


class OpenAIEmbedder(EmbedderPort):
    dim: int  # required by the port

    def __init__(
        self,
        model: str | None = None,
        *,
        settings_obj: Settings | None = None,
    ):
        configured_settings = settings_obj or settings
        self.model = model or configured_settings.openai_embedding_model
        # Default to a widely used OpenAI embedding dimensionality when unknown.
        # Don't depend on import-time settings for this fallback.
        self.dim = _MODEL_DIM.get(self.model, 1536)
        if not configured_settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required to use OpenAI embeddings.")
        self.client = create_openai_client(
            api_key=configured_settings.openai_api_key,
            timeout=configured_settings.openai_request_timeout,
            client_factory=OpenAI,
        )

    def embed(self, texts: Sequence[str]) -> Sequence[Embedding]:
        if not texts:
            # Avoid calling the API with empty input (some providers reject it).
            return []
        try:
            resp = self.client.embeddings.create(model=self.model, input=list(texts))
        except Exception as err:
            # Robust to different SDK exception classes
            raise RuntimeError(
                f"OpenAI embeddings error: {getattr(err, 'message', str(err))}"
            ) from err
        return [item.embedding for item in resp.data]

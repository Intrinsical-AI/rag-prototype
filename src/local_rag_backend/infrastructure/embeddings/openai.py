# src/infrastructure/embeddings/openai.py
"""
OpenAI embeddings implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from openai import OpenAI

from local_rag_backend.core.ports import EmbedderPort, Embedding
from local_rag_backend.settings import settings

_MODEL_DIM: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

DEFAULT_MODEL = settings.openai_embedding_model
DEFAULT_DIM = _MODEL_DIM.get(DEFAULT_MODEL, 1536)


class OpenAIEmbedder(EmbedderPort):
    dim: int  # required by the port

    def __init__(self, model: str | None = None):
        self.model = model or settings.openai_embedding_model
        self.dim = _MODEL_DIM.get(self.model, DEFAULT_DIM)
        self.client = OpenAI(api_key=settings.openai_api_key)

    def embed(self, texts: Sequence[str]) -> Sequence[Embedding]:
        try:
            resp = self.client.embeddings.create(model=self.model, input=list(texts))
        except Exception as err:
            # Robust to different SDK exception classes
            raise RuntimeError(
                f"OpenAI embeddings error: {getattr(err, 'message', str(err))}"
            ) from err
        return [item.embedding for item in resp.data]

"""
OpenAI embeddings implementation.
"""

from __future__ import annotations

from typing import Sequence

from openai import APIError, OpenAI  # type: ignore

from src.core.ports import EmbedderPort, Embedding
from src.settings import settings

_MODEL_DIM: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

DEFAULT_MODEL = settings.openai_embedding_model
DEFAULT_DIM = _MODEL_DIM.get(DEFAULT_MODEL, 1536)


class OpenAIEmbedder(EmbedderPort):
    dim: int  # requerido por el puerto

    def __init__(self, model: str | None = None):
        self.model = model or settings.openai_embedding_model
        self.dim = _MODEL_DIM.get(self.model, DEFAULT_DIM)
        self.client = OpenAI(api_key=settings.openai_api_key)

    def embed(self, texts: Sequence[str]) -> Sequence[Embedding]:
        try:
            resp = self.client.embeddings.create(model=self.model, input=list(texts))
        except APIError:
            raise
        return [item.embedding for item in resp.data]

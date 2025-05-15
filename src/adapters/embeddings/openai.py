from __future__ import annotations

"""OpenAI embeddings adapter (v1 API)
"""

from typing import List

from openai import OpenAI, APIError  # type: ignore

from src.core.ports import EmbedderPort
from src.settings import settings

__all__ = ["OpenAIEmbedder"]

_MODEL_DIM: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

DEFAULT_MODEL = settings.openai_embedding_model
DEFAULT_DIM = _MODEL_DIM.get(DEFAULT_MODEL, 1536)


class OpenAIEmbedder(EmbedderPort):
    """Adapter para la API de *embeddings* de OpenAI v1.x"""

    def __init__(self, *, model: str | None = None) -> None:
        self.model = model or settings.openai_embedding_model
        self.DIM: int = _MODEL_DIM.get(self.model, DEFAULT_DIM)

        # OpenAI v1+ Client
        self.client = OpenAI(api_key=settings.openai_api_key)

    # ------------------------------------------------------------------
    # Port Implementation
    # ------------------------------------------------------------------
    def embed(self, text: str) -> List[float]:
        """Devuelve el embedding como *lista* de floats."""
        try:
            resp = self.client.embeddings.create(model=self.model, input=text)
        except APIError as e:
            raise

        emb_vector = resp.data[0].embedding
        return emb_vector

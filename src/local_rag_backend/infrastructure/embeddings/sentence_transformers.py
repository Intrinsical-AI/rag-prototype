# src/infrastructure/embeddings/sentence_transformers.py
"""
SentenceTransformer embedder (CPU-friendly).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from sentence_transformers import SentenceTransformer

from local_rag_backend.core.ports import EmbedderPort

Embedding = Sequence[float]


class SentenceTransformerEmbedder(EmbedderPort):
    """Embedder using the sentence-transformers library."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = cast("int", self.model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> Sequence[Embedding]:
        """Embeds a sequence of texts into a sequence of vector embeddings."""
        if not texts:
            return []
        embeddings = self.model.encode(list(texts))
        return cast("Sequence[Embedding]", embeddings.tolist())

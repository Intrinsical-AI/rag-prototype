"""
File: src/infrastructure/embeddings/sentence_transformers.py
SentenceTransformer embedder (CPU-friendly).
"""

from collections.abc import Sequence
from typing import cast

from sentence_transformers import SentenceTransformer

from local_rag_backend.core.ports import EmbedderPort

Embedding = Sequence[float]


class SentenceTransformerEmbedder(EmbedderPort):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: Sequence[str]) -> Sequence[Embedding]:
        embeddings = self.model.encode(list(texts))
        return cast("Sequence[Embedding]", embeddings.tolist())

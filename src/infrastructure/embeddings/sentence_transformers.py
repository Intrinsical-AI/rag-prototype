"""
File: src/adapters/embeddings/sentence_transformers.p
SentenceTransformer embedder (CPU-friendly).
"""

from typing import Sequence

from sentence_transformers import SentenceTransformer

from src.core.ports import EmbedderPort

Embedding = Sequence[float]


class SentenceTransformerEmbedder(EmbedderPort):

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = 384

    def embed(self, texts: Sequence[str]) -> Sequence[Embedding]:
        return self.model.encode(list(texts)).tolist()

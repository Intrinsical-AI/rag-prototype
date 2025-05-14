"""
File: src/adapters/embeddings/sentence_transformers.p
SentenceTransformer embedder (CPU-friendly).
"""
from typing import List
from sentence_transformers import SentenceTransformer

from src.core.ports import EmbedderPort

class SentenceTransformerEmbedder(EmbedderPort):
    DIM = 384

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    # ------------------------------------------------------------------ #
    def embed(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()
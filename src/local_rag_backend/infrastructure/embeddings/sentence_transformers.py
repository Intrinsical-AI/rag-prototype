"""
Intrinsical-AI RAG Prototype
Copyright (c) 2025 Intrinsical-AI

Module: Sentence Transformers Embedder
Purpose: Text-to-vector embedding using Sentence Transformers library.
         Provides CPU-friendly semantic embeddings for document retrieval.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from sentence_transformers import SentenceTransformer

from local_rag_backend.core.ports import EmbedderPort

Embedding = Sequence[float]


class SentenceTransformerEmbedder(EmbedderPort):
    """CPU-friendly text embedder using Sentence Transformers.

    This embedder provides semantic vector representations of text using
    pre-trained transformer models. Optimized for CPU inference with
    good balance between speed and quality.

    Default model 'all-MiniLM-L6-v2' provides 384-dimensional embeddings
    with excellent performance for most RAG applications.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedder with specified model.

        Args:
            model_name: HuggingFace model identifier for sentence transformers
        """
        self.model = SentenceTransformer(model_name)
        self.dim = cast("int", self.model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> Sequence[Embedding]:
        """Convert text sequences to vector embeddings.

        Args:
            texts: Sequence of text strings to embed

        Returns:
            Sequence of vector embeddings as float lists

        Note:
            Returns empty list if no texts provided. Uses CPU inference
            for broad compatibility across deployment environments.
        """
        if not texts:
            return []
        embeddings = self.model.encode(list(texts))
        return cast("Sequence[Embedding]", embeddings.tolist())

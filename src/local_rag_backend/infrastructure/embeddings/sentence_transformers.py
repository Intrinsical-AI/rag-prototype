# src/infrastructure/embeddings/sentence_transformers.py
"""
SentenceTransformer embedder (CPU-friendly).
"""

from __future__ import annotations

import hashlib
import os
import random
import time
from collections.abc import Sequence
from typing import cast

from local_rag_backend.core.errors import LLMResponseError, LLMTimeoutError
from local_rag_backend.core.ports import EmbedderPort

Embedding = Sequence[float]


class SentenceTransformerEmbedder(EmbedderPort):
    """Embedder using the sentence-transformers library."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._synthetic = str(os.getenv("RAG_SYNTHETIC_EMBEDDINGS", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        # nosec S311: intentional non-cryptographic RNG for synthetic stress-mode jitter/failures.
        self._rng = random.Random()  # noqa: S311
        self._fail_rate = max(
            0.0, min(1.0, float(os.getenv("RAG_SYNTHETIC_EMBEDDING_FAIL_RATE", "0.0")))
        )
        self._jitter_min_ms = max(
            0.0, float(os.getenv("RAG_SYNTHETIC_EMBEDDING_JITTER_MIN_MS", "0.0"))
        )
        self._jitter_max_ms = max(
            self._jitter_min_ms,
            float(os.getenv("RAG_SYNTHETIC_EMBEDDING_JITTER_MAX_MS", str(self._jitter_min_ms))),
        )
        self.dim = int(os.getenv("RAG_SYNTHETIC_EMBEDDING_DIM", "384"))
        self.model = None

        if self._synthetic:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed. Install the 'dense-st' extra "
                "(e.g. `uv sync --extra dense-st`) or switch RETRIEVAL_MODE=sparse."
            ) from e

        self.model = SentenceTransformer(model_name)
        self.dim = cast("int", self.model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> Sequence[Embedding]:
        """Embeds a sequence of texts into a sequence of vector embeddings."""
        if not texts:
            return []
        if self._synthetic:
            if self._jitter_max_ms > 0.0:
                jitter_ms = self._rng.uniform(self._jitter_min_ms, self._jitter_max_ms)
                time.sleep(jitter_ms / 1000.0)
            if self._rng.random() < self._fail_rate:
                # Simulate upstream 502/504 behavior for dense-path stress testing.
                if self._rng.random() < 0.5:
                    raise LLMResponseError("Synthetic embeddings upstream failure (simulated 502)")
                raise LLMTimeoutError("Synthetic embeddings upstream timeout (simulated 504)")
            return [self._synthetic_vector(text) for text in texts]
        if self.model is None:  # pragma: no cover
            raise RuntimeError("SentenceTransformer model is not initialized.")
        embeddings = self.model.encode(list(texts))
        return cast("Sequence[Embedding]", embeddings.tolist())

    def _synthetic_vector(self, text: str) -> list[float]:
        seed = int.from_bytes(
            hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()[:8], "big"
        )
        # nosec S311: deterministic synthetic vectors for reproducible stress testing.
        rng = random.Random(seed)  # noqa: S311
        return [rng.uniform(-1.0, 1.0) for _ in range(int(self.dim))]

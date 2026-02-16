"""
Deterministic reranking helpers.

This is intentionally dependency-free. It exists to improve retrieval quality cheaply and
to provide a measurable knob for offline evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from local_rag_backend.core.services.text_processing import preprocess_text

if TYPE_CHECKING:
    from collections.abc import Sequence

    from local_rag_backend.core.domain.entities import Document
    from local_rag_backend.core.ports import RetrieverPort


def _tokens(text: str) -> set[str]:
    # Keep tokenization aligned with the sparse retriever (preprocess_text + \\w+).
    return set(re.findall(r"\w+", preprocess_text(text)))


@dataclass(frozen=True)
class OverlapV1Reranker:
    """
    Token overlap reranker (cheap heuristic).

    Score: |tokens(query) ∩ tokens(doc)| / max(1, |tokens(query)|)
    """

    def score(self, *, query: str, doc_text: str) -> float:
        qt = _tokens(query)
        if not qt:
            return 0.0
        dt = _tokens(doc_text)
        if not dt:
            return 0.0
        return float(len(qt & dt) / max(1, len(qt)))


class RerankingRetriever:
    """Wraps another RetrieverPort, reranking its top-N candidates before returning top-k."""

    def __init__(
        self,
        base: RetrieverPort,
        *,
        candidate_k: int,
        strategy: str = "overlap_v1",
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        self.base = base
        self.candidate_k = int(candidate_k)
        if strategy != "overlap_v1":
            raise ValueError(f"Unsupported reranker strategy: {strategy}")
        self._reranker = OverlapV1Reranker()

    def retrieve(self, query: str, k: int = 5) -> tuple[Sequence[Document], Sequence[float]]:
        if k <= 0:
            return [], []
        cand_k = max(int(k), int(self.candidate_k))
        docs, _scores = self.base.retrieve(query, cand_k)
        if not docs:
            return [], []

        scored = [(doc, self._reranker.score(query=query, doc_text=doc.content)) for doc in docs]
        # Sort by reranker score, stable on the original order (Python sort is stable).
        scored.sort(key=lambda t: t[1], reverse=True)

        top = scored[:k]
        if not top:
            return [], []

        docs_out = [d for d, _ in top]
        scores_out = [float(s) for _, s in top]
        return docs_out, scores_out

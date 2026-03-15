"""
Deterministic reranking helpers.

This is intentionally dependency-free. It exists to improve retrieval quality cheaply and
to provide a measurable knob for offline evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from local_rag_backend.core.domain.retrieval import (
    RetrievalRequest,
    RetrievalResult,
    RetrievedDoc,
)

if TYPE_CHECKING:
    from local_rag_backend.core.ports import RetrieverPort


def _tokens(text: str) -> set[str]:
    from local_rag_backend.core.services.text_processing import preprocess_text

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

    def retrieve(self, query: RetrievalRequest) -> RetrievalResult:
        request = query
        if request.top_k <= 0:
            return RetrievalResult(items=(), mode_used=request.mode, backend_used="reranked")
        cand_k = max(int(request.top_k), int(self.candidate_k))
        base_request = RetrievalRequest(
            query=request.query,
            top_k=cand_k,
            mode=request.mode,
            filters=request.filters,
            candidate_k=request.candidate_k,
            dual_candidate_k=request.dual_candidate_k,
            min_score=request.min_score,
        )
        structured_result = self.base.retrieve(base_request)
        docs = list(structured_result.documents)
        mode_used = structured_result.mode_used
        backend_used = structured_result.backend_used
        if not docs:
            return RetrievalResult(items=(), mode_used=mode_used, backend_used=backend_used)
        scored = [
            RetrievedDoc(
                document=doc,
                score=self._reranker.score(query=request.query, doc_text=doc.content),
                stage="reranker",
            )
            for doc in docs
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return RetrievalResult(
            items=tuple(scored[: request.top_k]),
            mode_used=mode_used,
            backend_used=backend_used,
            candidate_count=len(docs),
        )

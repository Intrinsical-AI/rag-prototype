from __future__ import annotations

import math

import pytest

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.domain.retrieval import RetrievalRequest, RetrievalResult, RetrievedDoc
from local_rag_backend.infrastructure.retrieval.hybrid import HybridRetriever


class _StaticRetriever:
    def __init__(self, *, docs: list[Document], scores: list[float], backend_used: str) -> None:
        self._docs = docs
        self._scores = scores
        self._backend_used = backend_used

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        return RetrievalResult(
            items=tuple(
                RetrievedDoc(document=doc, score=score, stage="test")
                for doc, score in zip(self._docs[: request.top_k], self._scores, strict=False)
            ),
            mode_used="sparse",
            backend_used=self._backend_used,
        )


def test_hybrid_rejects_scores_outside_normalized_range() -> None:
    dense = _StaticRetriever(
        docs=[Document(id=1, content="dense")],
        scores=[1.2],
        backend_used="dense-test",
    )
    sparse = _StaticRetriever(
        docs=[Document(id=2, content="sparse")],
        scores=[0.5],
        backend_used="sparse-test",
    )
    hybrid = HybridRetriever(dense=dense, sparse=sparse, alpha=0.5)

    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        hybrid.retrieve(RetrievalRequest(query="q", top_k=1, mode="hybrid"))


@pytest.mark.parametrize("bad_score", [math.nan, math.inf, -math.inf])
def test_hybrid_rejects_non_finite_scores(bad_score: float) -> None:
    dense = _StaticRetriever(
        docs=[Document(id=1, content="dense")],
        scores=[0.5],
        backend_used="dense-test",
    )
    sparse = _StaticRetriever(
        docs=[Document(id=2, content="sparse")],
        scores=[bad_score],
        backend_used="sparse-test",
    )
    hybrid = HybridRetriever(dense=dense, sparse=sparse, alpha=0.5)

    with pytest.raises(ValueError, match="non-finite score"):
        hybrid.retrieve(RetrievalRequest(query="q", top_k=1, mode="hybrid"))

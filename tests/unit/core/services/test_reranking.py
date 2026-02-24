# tests/unit/core/services/test_reranking.py

import pytest

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.services.reranking import RerankingRetriever


class _DummyRetriever:
    def __init__(self, docs):
        self._docs = docs

    def retrieve(self, query: str, k: int = 5):
        # Return docs in the given order, same score for all.
        return self._docs[:k], [0.1] * min(k, len(self._docs))


def test_overlap_reranker_moves_more_relevant_doc_first():
    docs = [
        Document(id=1, content="zzz zzz zzz", external_id="d1"),
        Document(id=2, content="capital france paris", external_id="d2"),
    ]
    base = _DummyRetriever(docs)
    rr = RerankingRetriever(base, candidate_k=10, strategy="overlap_v1")
    out_docs, out_scores = rr.retrieve("capital of france", 2)
    assert [d.external_id for d in out_docs] == ["d2", "d1"]
    assert out_scores[0] >= out_scores[1]


def test_reranking_retriever_rejects_invalid_strategy():
    with pytest.raises(ValueError, match="Unsupported reranker strategy"):
        RerankingRetriever(_DummyRetriever([]), candidate_k=3, strategy="nope")


def test_reranking_retriever_candidate_k_must_be_positive():
    with pytest.raises(ValueError, match="candidate_k must be positive"):
        RerankingRetriever(_DummyRetriever([]), candidate_k=0, strategy="overlap_v1")


def test_reranking_retriever_k_zero_returns_empty():
    rr = RerankingRetriever(_DummyRetriever([]), candidate_k=3, strategy="overlap_v1")
    docs, scores = rr.retrieve("q", 0)
    assert docs == []
    assert scores == []

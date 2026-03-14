# tests/unit/core/services/test_reranking.py

import pytest

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.domain.retrieval import RetrievalRequest, RetrievalResult, RetrievedDoc
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


def test_reranking_retriever_structured_request_preserves_metadata_and_candidate_floor():
    seen = {}
    docs = [
        Document(id=1, content="capital france paris", external_id="d1"),
        Document(id=2, content="zzz zzz zzz", external_id="d2"),
    ]

    class _StructuredRetriever:
        def retrieve(self, request):
            seen["request"] = request
            return RetrievalResult(
                items=tuple(
                    RetrievedDoc(document=doc, score=0.1, stage="base") for doc in reversed(docs)
                ),
                mode_used="dual",
                backend_used="elastic",
                candidate_count=7,
            )

    rr = RerankingRetriever(_StructuredRetriever(), candidate_k=4, strategy="overlap_v1")
    result = rr.retrieve(
        RetrievalRequest(
            query="capital france",
            top_k=2,
            mode="dual",
            filters=(),
            candidate_k=1,
            dual_candidate_k=3,
            min_score=0.2,
        )
    )

    assert seen["request"].top_k == 4
    assert seen["request"].candidate_k == 1
    assert seen["request"].dual_candidate_k == 3
    assert seen["request"].min_score == 0.2
    assert [item.document.external_id for item in result.items] == ["d1", "d2"]
    assert result.mode_used == "dual"
    assert result.backend_used == "elastic"
    assert result.candidate_count == 2


def test_reranking_retriever_structured_request_handles_legacy_base_results():
    docs = [
        Document(id=1, content="capital france paris", external_id="d1"),
        Document(id=2, content="zzz zzz zzz", external_id="d2"),
    ]

    class _LegacyRetriever:
        def retrieve(self, request):
            assert isinstance(request, RetrievalRequest)
            return docs, [0.1, 0.1]

    rr = RerankingRetriever(_LegacyRetriever(), candidate_k=2, strategy="overlap_v1")
    result = rr.retrieve(RetrievalRequest(query="capital france", top_k=1, mode="sparse"))

    assert [item.document.external_id for item in result.items] == ["d1"]
    assert result.backend_used == "legacy"


def test_reranking_retriever_structured_request_returns_empty_when_base_is_empty():
    class _EmptyRetriever:
        def retrieve(self, request):
            return RetrievalResult(items=(), mode_used=request.mode, backend_used="base")

    rr = RerankingRetriever(_EmptyRetriever(), candidate_k=2, strategy="overlap_v1")
    result = rr.retrieve(RetrievalRequest(query="capital france", top_k=1, mode="sparse"))

    assert result.items == ()
    assert result.mode_used == "sparse"
    assert result.backend_used == "base"

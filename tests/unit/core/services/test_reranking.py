# tests/unit/core/services/test_reranking.py

import pytest

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.domain.retrieval import RetrievalRequest, RetrievalResult, RetrievedDoc
from local_rag_backend.core.domain.types import DocId
from local_rag_backend.core.services.reranking import RerankingRetriever


class _DummyRetriever:
    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        return RetrievalResult(
            items=tuple(
                RetrievedDoc(document=doc, score=0.1, stage="base")
                for doc in self._docs[: request.top_k]
            ),
            mode_used=request.mode,
            backend_used="base",
            candidate_count=min(request.top_k, len(self._docs)),
        )


def test_overlap_reranker_moves_more_relevant_doc_first() -> None:
    docs = [
        Document(id=DocId("1"), content="zzz zzz zzz", external_id="d1"),
        Document(id=DocId("2"), content="capital france paris", external_id="d2"),
    ]
    base = _DummyRetriever(docs)
    rr = RerankingRetriever(base, candidate_k=10, strategy="overlap_v1")
    result = rr.retrieve(RetrievalRequest(query="capital of france", top_k=2, mode="sparse"))
    assert [item.document.external_id for item in result.items] == ["d2", "d1"]
    assert result.items[0].score >= result.items[1].score


def test_reranking_retriever_rejects_invalid_strategy() -> None:
    with pytest.raises(ValueError, match="Unsupported reranker strategy"):
        RerankingRetriever(_DummyRetriever([]), candidate_k=3, strategy="nope")


def test_reranking_retriever_candidate_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="candidate_k must be positive"):
        RerankingRetriever(_DummyRetriever([]), candidate_k=0, strategy="overlap_v1")


def test_retrieval_request_rejects_top_k_zero() -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        RetrievalRequest(query="q", top_k=0, mode="sparse")


def test_reranking_retriever_structured_request_preserves_metadata_and_candidate_floor() -> None:
    seen: dict[str, RetrievalRequest] = {}
    docs = [
        Document(id=DocId("1"), content="capital france paris", external_id="d1"),
        Document(id=DocId("2"), content="zzz zzz zzz", external_id="d2"),
    ]

    class _StructuredRetriever:
        def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
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


def test_reranking_retriever_structured_request_returns_empty_when_base_is_empty() -> None:
    class _EmptyRetriever:
        def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
            return RetrievalResult(items=(), mode_used=request.mode, backend_used="base")

    rr = RerankingRetriever(_EmptyRetriever(), candidate_k=2, strategy="overlap_v1")
    result = rr.retrieve(RetrievalRequest(query="capital france", top_k=1, mode="sparse"))

    assert result.items == ()
    assert result.mode_used == "sparse"
    assert result.backend_used == "base"

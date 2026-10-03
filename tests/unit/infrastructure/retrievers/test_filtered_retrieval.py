from dataclasses import replace

import pytest

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.domain.retrieval import (
    RetrievalFilter,
    RetrievalRequest,
    RetrievalResult,
    RetrievedDoc,
)
from local_rag_backend.core.domain.types import DocId
from local_rag_backend.infrastructure.retrieval.dense_vector import DenseVectorRetriever
from local_rag_backend.infrastructure.retrieval.hybrid import HybridRetriever
from local_rag_backend.infrastructure.search_backends.local_split import LocalSplitSearchRetriever


class Documents:
    def __init__(self):
        self.docs = [
            Document(
                DocId(str(i)), f"text {i}", metadata={"scope": "public" if i == 4 else "other"}
            )
            for i in range(1, 6)
        ]
        self.calls = []

    def get(self, ids):
        self.calls.append(list(ids))
        return [doc for doc in self.docs if doc.id in ids]

    def get_all_documents(self):
        return self.docs


class Embeddings:
    dim = 2

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


class Vectors:
    ntotal = 5

    def __init__(self):
        self.calls = []

    def similar(self, vector, k):
        self.calls.append(k)
        # Deliberately different scores between windows, as with min-max normalization.
        return [(DocId(str(i)), 1 - (i - 1) / max(1, k - 1)) for i in range(1, min(k, 5) + 1)]


def test_dense_overfetches_for_filters_once_per_document_and_embedding():
    docs, vectors, embedder = Documents(), Vectors(), Embeddings()
    request = RetrievalRequest(
        query="query", top_k=1, mode="dense", filters=(RetrievalFilter("scope", ("public",)),)
    )
    result = DenseVectorRetriever(embedder, vectors, docs).retrieve(request)

    assert [doc.id for doc in result.documents] == ["4"]
    assert result.scores == (0.0,)  # Score from the final window, not a previous one.
    assert result.candidate_count == 4
    assert vectors.calls == [1, 2, 4]
    assert docs.calls == [["1"], ["2"], ["3", "4"]]
    assert embedder.calls == [["query"]]


def test_dense_stops_at_exhaustion_without_matches():
    docs, vectors, embedder = Documents(), Vectors(), Embeddings()
    request = RetrievalRequest(
        query="query", top_k=1, mode="dense", filters=(RetrievalFilter("scope", ("absent",)),)
    )
    result = DenseVectorRetriever(embedder, vectors, docs).retrieve(request)
    assert result.items == ()
    assert vectors.calls == [1, 2, 4, 5]
    assert result.candidate_count == 5


def test_dense_uses_final_window_scores_after_filter_overfetch():
    docs, vectors, embedder = Documents(), Vectors(), Embeddings()
    docs.docs[1] = replace(docs.docs[1], metadata={"scope": "public"})
    result = DenseVectorRetriever(embedder, vectors, docs).retrieve(
        RetrievalRequest(
            query="query",
            top_k=2,
            mode="dense",
            filters=(RetrievalFilter("scope", ("public",)),),
        )
    )
    # ID 2 initially has score 0 in the two-item window, then 2/3 in the final four.
    assert [doc.id for doc in result.documents] == ["2", "4"]
    assert result.scores == pytest.approx((2 / 3, 0.0))
    assert vectors.calls == [2, 4]
    assert result.candidate_count == 4


def test_dense_does_not_overfetch_for_unfiltered_queries():
    docs, vectors, embedder = Documents(), Vectors(), Embeddings()
    request = RetrievalRequest(query="query", top_k=2, mode="dense")
    result = DenseVectorRetriever(embedder, vectors, docs).retrieve(request)
    assert [doc.id for doc in result.documents] == ["1", "2"]
    assert vectors.calls == [2]


def test_local_dense_and_hybrid_share_filter_semantics():
    docs, vectors, embedder = Documents(), Vectors(), Embeddings()
    local = LocalSplitSearchRetriever(doc_repo=docs, embedder=embedder, vector_repo=vectors)
    request = RetrievalRequest(
        query="text", top_k=1, mode="dense", filters=(RetrievalFilter("scope", ("public",)),)
    )
    assert [doc.id for doc in local.retrieve(request).documents] == ["4"]
    hybrid = HybridRetriever(dense=local, sparse=local, alpha=0.5)
    assert [doc.id for doc in hybrid.retrieve(replace(request, mode="hybrid")).documents] == ["4"]


def test_hybrid_preserves_filters_after_fusion():
    requests = []
    doc = Document(DocId("1"), "text")

    class Branch:
        def __init__(self, score):
            self.score = score

        def retrieve(self, request):
            requests.append(request)
            return RetrievalResult(
                items=(RetrievedDoc(doc, self.score),), mode_used=request.mode, backend_used="test"
            )

    filters = (RetrievalFilter("scope", ("public",)),)
    request = RetrievalRequest(query="text", top_k=1, mode="hybrid", filters=filters)
    hybrid = HybridRetriever(dense=Branch(0.4), sparse=Branch(1.0), alpha=0.5)
    result = hybrid.retrieve(request)
    assert result.scores == pytest.approx((0.7,))
    assert [item.mode for item in requests] == ["dense", "sparse"]
    assert all(item.filters == filters for item in requests)


def test_sparse_tied_rankings_have_consistent_prefixes():
    docs = Documents()
    docs.docs = [Document(DocId("1"), "alpha"), Document(DocId("2"), "beta")]
    sparse = LocalSplitSearchRetriever(doc_repo=docs)
    one = sparse.retrieve(RetrievalRequest(query="beta", top_k=1))
    two = sparse.retrieve(RetrievalRequest(query="beta", top_k=2))
    assert one.documents[0].id == two.documents[0].id == "2"

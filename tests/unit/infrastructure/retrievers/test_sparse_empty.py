"""
Test sparse retriever behavior with empty corpus.
"""

from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever


class MockDocumentRepo:
    """Mock document repository that returns empty results."""

    def get(self, ids):
        return []


def test_sparse_empty_corpus_returns_empty():
    """Test that sparse retriever with empty corpus returns empty results."""
    retriever = SparseBM25Retriever(documents=[], doc_ids=[], doc_repo=MockDocumentRepo())

    docs, scores = retriever.retrieve("any query", k=3)

    assert docs == []
    assert scores == []


def test_sparse_empty_query_returns_empty():
    """Test that sparse retriever with empty query returns empty results."""
    retriever = SparseBM25Retriever(
        documents=["sample document"], doc_ids=[1], doc_repo=MockDocumentRepo()
    )

    docs, scores = retriever.retrieve("", k=3)

    assert docs == []
    assert scores == []


def test_sparse_zero_k_returns_empty():
    """Test that sparse retriever with k=0 returns empty results."""
    retriever = SparseBM25Retriever(
        documents=["sample document"], doc_ids=[1], doc_repo=MockDocumentRepo()
    )

    docs, scores = retriever.retrieve("query", k=0)

    assert docs == []
    assert scores == []

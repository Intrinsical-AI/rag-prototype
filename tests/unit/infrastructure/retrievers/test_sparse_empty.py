"""
Test sparse retriever behavior with empty corpus.
"""

import pytest

from local_rag_backend.core.domain.retrieval import RetrievalRequest
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever


class MockDocumentRepo:
    """Mock document repository that returns empty results."""

    def get(self, ids):
        return []


def test_sparse_empty_corpus_returns_empty():
    """Test that sparse retriever with empty corpus returns empty results."""
    retriever = SparseBM25Retriever(documents=[], doc_ids=[], doc_repo=MockDocumentRepo())

    result = retriever.retrieve(RetrievalRequest(query="any query", top_k=3, mode="sparse"))

    assert result.documents == ()
    assert result.scores == ()


def test_sparse_empty_query_raises():
    """RetrievalRequest rejects blank queries at construction time."""
    with pytest.raises(ValueError, match="query must not be blank"):
        RetrievalRequest(query="", top_k=3, mode="sparse")


def test_sparse_zero_k_raises():
    """RetrievalRequest rejects top_k=0 at construction time."""
    with pytest.raises(ValueError, match="top_k must be positive"):
        RetrievalRequest(query="query", top_k=0, mode="sparse")

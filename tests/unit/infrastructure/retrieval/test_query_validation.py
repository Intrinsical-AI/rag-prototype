"""
Tests for query validation in retrievers.

This module tests the critical bug fixes for query validation
to prevent IndexError and other crashes from invalid queries.
"""

from unittest.mock import Mock

import pytest

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.infrastructure.retrieval.dense_faiss import DenseFaissRetriever
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever


class TestQueryValidation:
    """Test query validation and edge cases in retrievers."""

    @pytest.fixture
    def mock_doc_repo(self):
        """Mock document repository."""
        repo = Mock()
        repo.get.return_value = [
            Document(id=1, content="First document"),
            Document(id=2, content="Second document"),
        ]
        return repo

    @pytest.fixture
    def mock_embedder(self):
        """Mock embedder."""
        embedder = Mock()
        embedder.embed.return_value = [[0.1, 0.2, 0.3]]
        return embedder

    @pytest.fixture
    def mock_faiss_index(self):
        """Mock FAISS index."""
        index = Mock()
        index.similar.return_value = [(1, 0.9), (2, 0.8)]
        return index

    @pytest.fixture
    def dense_retriever(self, mock_embedder, mock_faiss_index, mock_doc_repo):
        """Dense retriever with mocked dependencies."""
        return DenseFaissRetriever(mock_embedder, mock_faiss_index, mock_doc_repo)

    @pytest.fixture
    def sparse_retriever(self, mock_doc_repo):
        """Sparse retriever with mocked dependencies."""
        corpus = ["first document", "second document"]
        doc_ids = [1, 2]
        return SparseBM25Retriever(corpus, doc_ids, mock_doc_repo)

    @pytest.mark.parametrize(
        "invalid_query",
        [
            None,  # None query
            "",  # Empty string
            "   ",  # Whitespace only
            "\t\n",  # Tab and newline only
            "     \t   \n   ",  # Mixed whitespace
        ],
    )
    def test_dense_retriever_invalid_queries(self, dense_retriever, invalid_query):
        """Test dense retriever handles invalid queries gracefully."""
        docs, scores = dense_retriever.retrieve(invalid_query, k=5)

        assert docs == []
        assert scores == []
        assert len(docs) == len(scores)

    @pytest.mark.parametrize(
        "invalid_query",
        [
            None,  # None query
            "",  # Empty string
            "   ",  # Whitespace only
            "\t\n",  # Tab and newline only
            "     \t   \n   ",  # Mixed whitespace
        ],
    )
    def test_sparse_retriever_invalid_queries(self, sparse_retriever, invalid_query):
        """Test sparse retriever handles invalid queries gracefully."""
        docs, scores = sparse_retriever.retrieve(invalid_query, k=5)

        assert docs == []
        assert scores == []
        assert len(docs) == len(scores)

    def test_dense_retriever_embedder_returns_empty(
        self, mock_embedder, mock_faiss_index, mock_doc_repo
    ):
        """Test dense retriever when embedder returns empty list."""
        mock_embedder.embed.return_value = []  # Empty embeddings

        retriever = DenseFaissRetriever(mock_embedder, mock_faiss_index, mock_doc_repo)
        docs, scores = retriever.retrieve("valid query", k=5)

        assert docs == []
        assert scores == []
        # Should not call FAISS when no embeddings
        mock_faiss_index.similar.assert_not_called()

    def test_dense_retriever_embedder_failure(self, mock_embedder, mock_faiss_index, mock_doc_repo):
        """Test dense retriever when embedder fails."""
        mock_embedder.embed.side_effect = Exception("Embedding service down")

        retriever = DenseFaissRetriever(mock_embedder, mock_faiss_index, mock_doc_repo)

        with pytest.raises(RuntimeError, match="Dense retrieval failed for query"):
            retriever.retrieve("valid query", k=5)

    @pytest.mark.parametrize(
        "query,expected_normalized",
        [
            ("  hello world  ", "hello world"),
            ("\ttest query\n", "test query"),
            ("   mixed   spaces   ", "mixed   spaces"),
        ],
    )
    def test_dense_retriever_query_normalization(
        self, mock_embedder, mock_faiss_index, mock_doc_repo, query, expected_normalized
    ):
        """Test that queries are properly normalized before embedding."""
        retriever = DenseFaissRetriever(mock_embedder, mock_faiss_index, mock_doc_repo)
        retriever.retrieve(query, k=5)

        # Verify embedder was called with normalized query
        mock_embedder.embed.assert_called_once_with([expected_normalized])

    @pytest.mark.parametrize(
        "query,expected_normalized",
        [
            ("  hello world  ", "hello world"),
            ("\ttest query\n", "test query"),
            ("   mixed   spaces   ", "mixed   spaces"),
        ],
    )
    def test_sparse_retriever_query_normalization(self, mock_doc_repo, query, expected_normalized):
        """Test that queries are properly normalized in sparse retriever."""
        corpus = ["hello world document", "test query document"]
        doc_ids = [1, 2]

        retriever = SparseBM25Retriever(corpus, doc_ids, mock_doc_repo)

        # Mock the _tokenize method to capture the normalized query
        original_tokenize = retriever._tokenize
        retriever._tokenize = Mock(side_effect=original_tokenize)

        retriever.retrieve(query, k=5)

        # Verify tokenize was called with normalized query
        retriever._tokenize.assert_called_with(expected_normalized)

    @pytest.mark.parametrize("k_value", [0, -1, -10])
    def test_retrievers_handle_invalid_k(self, dense_retriever, sparse_retriever, k_value):
        """Test both retrievers handle invalid k values."""
        # Dense retriever
        docs, scores = dense_retriever.retrieve("valid query", k=k_value)
        assert docs == []
        assert scores == []

        # Sparse retriever
        docs, scores = sparse_retriever.retrieve("valid query", k=k_value)
        assert docs == []
        assert scores == []

    def test_dense_retriever_non_string_query_type_safety(self, dense_retriever):
        """Test dense retriever type safety with non-string queries."""
        # These should be handled gracefully by _is_valid_query
        non_string_queries = [123, [], {}, object()]

        for invalid_query in non_string_queries:
            docs, scores = dense_retriever.retrieve(invalid_query, k=5)  # type: ignore
            assert docs == []
            assert scores == []

    def test_sparse_retriever_non_string_query_type_safety(self, sparse_retriever):
        """Test sparse retriever type safety with non-string queries."""
        # These should be handled gracefully by _is_valid_query
        non_string_queries = [123, [], {}, object()]

        for invalid_query in non_string_queries:
            docs, scores = sparse_retriever.retrieve(invalid_query, k=5)  # type: ignore
            assert docs == []
            assert scores == []

    def test_dense_retriever_query_validation_method(self, dense_retriever):
        """Test the _is_valid_query method directly."""
        # Valid queries
        assert dense_retriever._is_valid_query("valid query") is True
        assert dense_retriever._is_valid_query("  trimmed  ") is True
        assert dense_retriever._is_valid_query("a") is True

        # Invalid queries
        assert dense_retriever._is_valid_query(None) is False
        assert dense_retriever._is_valid_query("") is False
        assert dense_retriever._is_valid_query("   ") is False
        assert dense_retriever._is_valid_query(123) is False  # type: ignore

    def test_sparse_retriever_query_validation_method(self, sparse_retriever):
        """Test the _is_valid_query method directly."""
        # Valid queries
        assert sparse_retriever._is_valid_query("valid query") is True
        assert sparse_retriever._is_valid_query("  trimmed  ") is True
        assert sparse_retriever._is_valid_query("a") is True

        # Invalid queries
        assert sparse_retriever._is_valid_query(None) is False
        assert sparse_retriever._is_valid_query("") is False
        assert sparse_retriever._is_valid_query("   ") is False
        assert sparse_retriever._is_valid_query(123) is False  # type: ignore

    def test_dense_retriever_error_context_preservation(
        self, mock_embedder, mock_faiss_index, mock_doc_repo
    ):
        """Test that error context is preserved in dense retriever."""
        mock_embedder.embed.side_effect = ValueError("Invalid embedding input")

        retriever = DenseFaissRetriever(mock_embedder, mock_faiss_index, mock_doc_repo)

        with pytest.raises(RuntimeError) as exc_info:
            retriever.retrieve("test query", k=5)

        error_msg = str(exc_info.value)
        assert "Dense retrieval failed for query 'test query'" in error_msg
        assert "Invalid embedding input" in str(exc_info.value.__cause__)

    def test_sparse_retriever_no_bm25_index(self, mock_doc_repo):
        """Test sparse retriever when no BM25 index is available."""
        # Create retriever with empty corpus (no BM25 index)
        retriever = SparseBM25Retriever([], [], mock_doc_repo)

        docs, scores = retriever.retrieve("valid query", k=5)

        assert docs == []
        assert scores == []

    def test_sparse_retriever_query_with_no_tokens(self, mock_doc_repo):
        """Test sparse retriever when query produces no tokens."""
        corpus = ["document with words"]
        doc_ids = [1]

        retriever = SparseBM25Retriever(corpus, doc_ids, mock_doc_repo)

        # Query with only punctuation/symbols that produce no tokens
        docs, scores = retriever.retrieve("!@#$%^&*()", k=5)

        assert docs == []
        assert scores == []

    def test_retrievers_consistency_with_valid_query(self, dense_retriever, sparse_retriever):
        """Test that both retrievers return consistent array lengths with valid queries."""
        # Dense retriever
        docs, scores = dense_retriever.retrieve("valid test query", k=3)
        assert len(docs) == len(scores)

        # Sparse retriever
        docs, scores = sparse_retriever.retrieve("valid test query", k=3)
        assert len(docs) == len(scores)

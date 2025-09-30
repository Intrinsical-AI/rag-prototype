"""
Tests for array consistency fixes in retrievers.

This module tests the critical bug fixes for array length mismatches
between documents and scores in retrieval operations.
"""

from unittest.mock import Mock

import pytest

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.infrastructure.retrieval.dense_faiss import DenseFaissRetriever
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever


class TestArrayConsistencyFixes:
    """Test array consistency between docs and scores in retrievers."""

    @pytest.fixture
    def mock_doc_repo(self):
        """Mock document repository."""
        return Mock()

    @pytest.fixture
    def mock_embedder(self):
        """Mock embedder."""
        embedder = Mock()
        embedder.embed.return_value = [[0.1, 0.2, 0.3]]  # Single embedding
        return embedder

    @pytest.fixture
    def mock_faiss_index(self):
        """Mock FAISS index."""
        return Mock()

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            Document(id=1, content="First document"),
            Document(id=2, content="Second document"),
            Document(id=3, content="Third document"),
        ]

    @pytest.mark.parametrize("missing_doc_ids,expected_count", [
        ([2], 2),  # One document missing
        ([1, 3], 1),  # Two documents missing
        ([1, 2, 3], 0),  # All documents missing
        ([], 3),  # No documents missing
    ])
    def test_dense_faiss_retriever_missing_documents(
        self, mock_embedder, mock_faiss_index, mock_doc_repo, sample_documents, missing_doc_ids, expected_count
    ):
        """Test dense retriever handles missing documents correctly."""
        # Setup: FAISS returns 3 results but some docs are missing from DB
        mock_faiss_index.similar.return_value = [
            (1, 0.9), (2, 0.8), (3, 0.7)
        ]

        # Filter out missing documents
        available_docs = [doc for doc in sample_documents if doc.id not in missing_doc_ids]
        mock_doc_repo.get.return_value = available_docs

        retriever = DenseFaissRetriever(mock_embedder, mock_faiss_index, mock_doc_repo)
        docs, scores = retriever.retrieve("test query", k=3)

        # Verify array consistency
        assert len(docs) == len(scores), f"Array length mismatch: {len(docs)} docs vs {len(scores)} scores"
        assert len(docs) == expected_count, f"Expected {expected_count} docs, got {len(docs)}"

        # Verify correct documents are returned
        returned_ids = [doc.id for doc in docs]
        expected_ids = [doc.id for doc in available_docs]
        assert returned_ids == expected_ids, f"Document IDs mismatch: {returned_ids} vs {expected_ids}"

    @pytest.mark.parametrize("missing_doc_ids,expected_count", [
        ([2], 2),  # One document missing
        ([1, 3], 1),  # Two documents missing
        ([1, 2, 3], 0),  # All documents missing
        ([], 3),  # No documents missing
    ])
    def test_sparse_bm25_retriever_missing_documents(
        self, mock_doc_repo, sample_documents, missing_doc_ids, expected_count
    ):
        """Test sparse retriever handles missing documents correctly."""
        # Setup: Create retriever with corpus
        corpus = ["first document", "second document", "third document"]
        doc_ids = [1, 2, 3]

        # Filter out missing documents
        available_docs = [doc for doc in sample_documents if doc.id not in missing_doc_ids]
        mock_doc_repo.get.return_value = available_docs

        retriever = SparseBM25Retriever(corpus, doc_ids, mock_doc_repo)
        docs, scores = retriever.retrieve("document", k=3)

        # Verify array consistency
        assert len(docs) == len(scores), f"Array length mismatch: {len(docs)} docs vs {len(scores)} scores"
        assert len(docs) == expected_count, f"Expected {expected_count} docs, got {len(docs)}"

        # Verify scores are properly filtered
        if docs:
            assert all(isinstance(score, float) for score in scores), "All scores should be floats"
            assert all(0.0 <= score <= 1.0 for score in scores), "All scores should be normalized [0,1]"

    def test_dense_faiss_empty_results(self, mock_embedder, mock_faiss_index, mock_doc_repo):
        """Test dense retriever with empty FAISS results."""
        mock_faiss_index.similar.return_value = []

        retriever = DenseFaissRetriever(mock_embedder, mock_faiss_index, mock_doc_repo)
        docs, scores = retriever.retrieve("test query", k=3)

        assert docs == []
        assert scores == []
        assert len(docs) == len(scores)

    def test_sparse_bm25_empty_corpus(self, mock_doc_repo):
        """Test sparse retriever with empty corpus."""
        retriever = SparseBM25Retriever([], [], mock_doc_repo)
        docs, scores = retriever.retrieve("test query", k=3)

        assert docs == []
        assert scores == []
        assert len(docs) == len(scores)

    def test_dense_faiss_database_error(self, mock_embedder, mock_faiss_index, mock_doc_repo):
        """Test dense retriever when database query fails."""
        mock_faiss_index.similar.return_value = [(1, 0.9), (2, 0.8)]
        mock_doc_repo.get.side_effect = Exception("Database error")

        retriever = DenseFaissRetriever(mock_embedder, mock_faiss_index, mock_doc_repo)

        with pytest.raises(Exception, match="Database error"):
            retriever.retrieve("test query", k=2)

    def test_sparse_bm25_score_consistency_order(self, mock_doc_repo, sample_documents):
        """Test that scores maintain correct order after filtering."""
        corpus = ["first document", "second document", "third document"]
        doc_ids = [1, 2, 3]

        # Only return documents 1 and 3 (skip 2)
        available_docs = [sample_documents[0], sample_documents[2]]  # docs 1 and 3
        mock_doc_repo.get.return_value = available_docs

        retriever = SparseBM25Retriever(corpus, doc_ids, mock_doc_repo)
        docs, scores = retriever.retrieve("document", k=3)

        # Should return 2 docs and 2 scores
        assert len(docs) == 2
        assert len(scores) == 2

        # Verify order is maintained (highest score first)
        if len(scores) > 1:
            assert scores[0] >= scores[1], "Scores should be in descending order"

    @pytest.mark.parametrize("k_value,expected_behavior", [
        (0, "empty_results"),
        (-1, "empty_results"),
        (1, "normal_operation"),
        (10, "normal_operation"),  # More than available
    ])
    def test_retriever_k_parameter_edge_cases(
        self, mock_embedder, mock_faiss_index, mock_doc_repo, sample_documents, k_value, expected_behavior
    ):
        """Test retriever behavior with various k values."""
        if expected_behavior == "empty_results":
            mock_faiss_index.similar.return_value = []
        else:
            # Simulate FAISS respecting k parameter
            all_results = [(1, 0.9), (2, 0.8), (3, 0.7)]
            limited_results = all_results[:max(0, k_value)]  # Respect k_value
            mock_faiss_index.similar.return_value = limited_results

        mock_doc_repo.get.return_value = sample_documents

        retriever = DenseFaissRetriever(mock_embedder, mock_faiss_index, mock_doc_repo)
        docs, scores = retriever.retrieve("test query", k=k_value)

        # Always ensure array consistency
        assert len(docs) == len(scores)

        if expected_behavior == "empty_results":
            assert len(docs) == 0
        else:
            expected_max = min(max(0, k_value), len(sample_documents))
            assert len(docs) <= expected_max

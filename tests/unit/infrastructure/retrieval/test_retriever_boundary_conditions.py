"""
Comprehensive boundary condition tests for all retrievers.

Tests critical edge cases that could cause silent failures:
- Empty queries and results
- Invalid input parameters
- Score normalization edge cases
- Document-score correspondence
- Resource exhaustion scenarios
"""

from unittest.mock import Mock

import pytest

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.infrastructure.retrieval.dense_faiss import DenseFaissRetriever
from local_rag_backend.infrastructure.retrieval.hybrid import HybridRetriever
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever


class TestDenseFaissRetrieverBoundaryConditions:
    """Test boundary conditions for DenseFaissRetriever."""

    @pytest.fixture
    def mock_embedder(self):
        """Mock embedder for testing."""
        embedder = Mock()
        embedder.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]
        return embedder

    @pytest.fixture
    def mock_faiss_index(self):
        """Mock FAISS index for testing."""
        index = Mock()
        index.similar.return_value = [(1, 0.8), (2, 0.6)]
        return index

    @pytest.fixture
    def mock_doc_repo(self):
        """Mock document repository for testing."""
        repo = Mock()
        repo.get.return_value = [Document(id=1, content="Doc 1"), Document(id=2, content="Doc 2")]
        return repo

    @pytest.fixture
    def retriever(self, mock_embedder, mock_faiss_index, mock_doc_repo):
        """Create DenseFaissRetriever for testing."""
        return DenseFaissRetriever(mock_embedder, mock_faiss_index, mock_doc_repo)

    @pytest.mark.parametrize(
        "query,expected_result",
        [
            (None, ([], [])),
            ("", ([], [])),
            ("   ", ([], [])),
            ("\t\n\r", ([], [])),
        ],
    )
    def test_retrieve_invalid_queries(self, retriever, query, expected_result):
        """Test retrieve with invalid query inputs."""
        result = retriever.retrieve(query, k=5)
        assert result == expected_result

    @pytest.mark.parametrize("k", [0, -1, -10])
    def test_retrieve_invalid_k_values(self, retriever, k):
        """Test retrieve with invalid k values."""
        result = retriever.retrieve("valid query", k=k)
        assert result == ([], [])

    def test_retrieve_embedder_returns_empty(self, retriever, mock_embedder):
        """Test behavior when embedder returns empty list."""
        mock_embedder.embed.return_value = []

        result = retriever.retrieve("query", k=5)
        assert result == ([], [])

    def test_retrieve_faiss_returns_empty(self, retriever, mock_faiss_index):
        """Test behavior when FAISS returns no results."""
        mock_faiss_index.similar.return_value = []

        result = retriever.retrieve("query", k=5)
        assert result == ([], [])

    def test_retrieve_doc_repo_missing_documents(self, retriever, mock_faiss_index, mock_doc_repo):
        """Test behavior when document repo is missing some documents."""
        # FAISS returns 3 documents
        mock_faiss_index.similar.return_value = [(1, 0.8), (2, 0.6), (3, 0.4)]

        # Doc repo only has 2 documents (missing ID 3)
        mock_doc_repo.get.return_value = [
            Document(id=1, content="Doc 1"),
            Document(id=2, content="Doc 2"),
        ]

        docs, scores = retriever.retrieve("query", k=5)

        # Should return only available documents with corresponding scores
        assert len(docs) == 2
        assert len(scores) == 2
        assert [doc.id for doc in docs] == [1, 2]
        assert scores == [0.8, 0.6]

    def test_retrieve_doc_repo_wrong_order(self, retriever, mock_faiss_index, mock_doc_repo):
        """Test that retriever maintains FAISS result order despite doc repo order."""
        # FAISS returns in specific order
        mock_faiss_index.similar.return_value = [(3, 0.9), (1, 0.7), (2, 0.5)]

        # Doc repo returns in different order
        mock_doc_repo.get.return_value = [
            Document(id=1, content="Doc 1"),
            Document(id=2, content="Doc 2"),
            Document(id=3, content="Doc 3"),
        ]

        docs, scores = retriever.retrieve("query", k=5)

        # Should maintain FAISS order
        assert [doc.id for doc in docs] == [3, 1, 2]
        assert scores == [0.9, 0.7, 0.5]

    def test_retrieve_embedder_exception(self, retriever, mock_embedder):
        """Test behavior when embedder raises exception."""
        mock_embedder.embed.side_effect = Exception("Embedding service down")

        with pytest.raises(RuntimeError, match="Dense retrieval failed for query"):
            retriever.retrieve("query", k=5)

    def test_retrieve_faiss_exception(self, retriever, mock_faiss_index):
        """Test behavior when FAISS raises exception."""
        mock_faiss_index.similar.side_effect = Exception("FAISS index corrupted")

        with pytest.raises(RuntimeError, match="Dense retrieval failed for query"):
            retriever.retrieve("query", k=5)

    def test_retrieve_large_k_value(self, retriever, mock_faiss_index):
        """Test retrieve with very large k value."""
        # FAISS returns fewer results than requested
        mock_faiss_index.similar.return_value = [(1, 0.8), (2, 0.6)]

        docs, scores = retriever.retrieve("query", k=1000)

        # Should return only available results
        assert len(docs) == 2
        assert len(scores) == 2


class TestSparseBM25RetrieverBoundaryConditions:
    """Test boundary conditions for SparseBM25Retriever."""

    @pytest.fixture
    def mock_doc_repo(self):
        """Mock document repository for testing."""
        repo = Mock()
        repo.get.return_value = [Document(id=1, content="Doc 1"), Document(id=2, content="Doc 2")]
        return repo

    def test_empty_corpus_initialization(self, mock_doc_repo):
        """Test initialization with empty corpus."""
        retriever = SparseBM25Retriever([], [], mock_doc_repo)

        # Should handle gracefully
        assert retriever.bm25 is None

        result = retriever.retrieve("query", k=5)
        assert result == ([], [])

    def test_whitespace_only_documents(self, mock_doc_repo):
        """Test initialization with whitespace-only documents."""
        documents = ["   ", "\t\n", ""]
        doc_ids = [1, 2, 3]

        retriever = SparseBM25Retriever(documents, doc_ids, mock_doc_repo)

        # Should handle gracefully (no valid tokens)
        result = retriever.retrieve("query", k=5)
        assert result == ([], [])

    def test_single_document_corpus(self, mock_doc_repo):
        """Test with single document corpus."""
        documents = ["single document"]
        doc_ids = [1]

        retriever = SparseBM25Retriever(documents, doc_ids, mock_doc_repo)

        # Should work with single document
        assert retriever.bm25 is not None

        docs, _scores = retriever.retrieve("document", k=5)
        assert len(docs) <= 1  # Can't return more than available

    @pytest.mark.parametrize("query", [None, "", "   ", "\t\n"])
    def test_invalid_queries(self, mock_doc_repo, query):
        """Test BM25 with invalid queries."""
        documents = ["doc1", "doc2"]
        doc_ids = [1, 2]

        retriever = SparseBM25Retriever(documents, doc_ids, mock_doc_repo)
        result = retriever.retrieve(query, k=5)
        assert result == ([], [])

    def test_query_with_no_matching_tokens(self, mock_doc_repo):
        """Test query that produces no matching tokens."""
        documents = ["hello world", "foo bar"]
        doc_ids = [1, 2]

        retriever = SparseBM25Retriever(documents, doc_ids, mock_doc_repo)

        # Query with only punctuation/special chars
        result = retriever.retrieve("!@#$%^&*()", k=5)
        assert result == ([], [])

    def test_score_normalization_identical_scores(self, mock_doc_repo):
        """Test score normalization when all BM25 scores are identical."""
        documents = ["same word", "same word", "same word"]
        doc_ids = [1, 2, 3]

        mock_doc_repo.get.return_value = [
            Document(id=1, content="same word"),
            Document(id=2, content="same word"),
            Document(id=3, content="same word"),
        ]

        retriever = SparseBM25Retriever(documents, doc_ids, mock_doc_repo)

        _docs, scores = retriever.retrieve("same", k=3)

        # All scores should be normalized to 1.0
        assert len(scores) == 3
        assert all(score == 1.0 for score in scores)

    def test_large_corpus_performance(self, mock_doc_repo):
        """Test with large corpus to check performance boundaries."""
        # Create large corpus
        corpus_size = 1000
        documents = [f"document number {i} with unique content" for i in range(corpus_size)]
        doc_ids = list(range(1, corpus_size + 1))

        # Mock doc repo to return subset
        mock_doc_repo.get.return_value = [Document(id=i, content=f"doc {i}") for i in range(1, 11)]

        retriever = SparseBM25Retriever(documents, doc_ids, mock_doc_repo)

        # Should handle large corpus
        docs, scores = retriever.retrieve("document", k=10)
        assert len(docs) <= 10
        assert len(scores) <= 10


class TestHybridRetrieverBoundaryConditions:
    """Test boundary conditions for HybridRetriever."""

    @pytest.fixture
    def mock_dense_retriever(self):
        """Mock dense retriever."""
        retriever = Mock()
        retriever.retrieve.return_value = (
            [Document(id=1, content="Dense 1"), Document(id=2, content="Dense 2")],
            [0.9, 0.7],
        )
        return retriever

    @pytest.fixture
    def mock_sparse_retriever(self):
        """Mock sparse retriever."""
        retriever = Mock()
        retriever.retrieve.return_value = (
            [Document(id=2, content="Sparse 2"), Document(id=3, content="Sparse 3")],
            [0.8, 0.6],
        )
        return retriever

    @pytest.mark.parametrize("alpha", [-0.1, 1.1, 2.0, -1.0])
    def test_invalid_alpha_values(self, mock_dense_retriever, mock_sparse_retriever, alpha):
        """Test HybridRetriever with invalid alpha values."""
        with pytest.raises(
            ValueError, match=r"Alpha for hybrid retrieval must be between 0\.0 and 1\.0"
        ):
            HybridRetriever(mock_dense_retriever, mock_sparse_retriever, alpha=alpha)

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_valid_alpha_values(self, mock_dense_retriever, mock_sparse_retriever, alpha):
        """Test HybridRetriever with valid alpha values."""
        # Should not raise exception
        retriever = HybridRetriever(mock_dense_retriever, mock_sparse_retriever, alpha=alpha)
        assert retriever.alpha == alpha

    def test_both_retrievers_return_empty(self, mock_dense_retriever, mock_sparse_retriever):
        """Test when both retrievers return empty results."""
        mock_dense_retriever.retrieve.return_value = ([], [])
        mock_sparse_retriever.retrieve.return_value = ([], [])

        retriever = HybridRetriever(mock_dense_retriever, mock_sparse_retriever)
        result = retriever.retrieve("query", k=5)

        assert result == ([], [])

    def test_one_retriever_returns_empty(self, mock_dense_retriever, mock_sparse_retriever):
        """Test when one retriever returns empty results."""
        mock_dense_retriever.retrieve.return_value = ([], [])
        mock_sparse_retriever.retrieve.return_value = ([Document(id=1, content="Sparse 1")], [0.8])

        retriever = HybridRetriever(mock_dense_retriever, mock_sparse_retriever, alpha=0.5)
        docs, scores = retriever.retrieve("query", k=5)

        # Should return sparse results with normalized scores
        assert len(docs) == 1
        assert len(scores) == 1
        assert docs[0].id == 1

    def test_score_normalization_edge_cases(self, mock_dense_retriever, mock_sparse_retriever):
        """Test score normalization with edge cases."""
        # Test with single scores from each retriever (normalized to 1.0)
        # But different documents, so hybrid fusion applies
        mock_dense_retriever.retrieve.return_value = ([Document(id=1, content="Dense 1")], [0.5])
        mock_sparse_retriever.retrieve.return_value = ([Document(id=2, content="Sparse 2")], [0.5])

        retriever = HybridRetriever(mock_dense_retriever, mock_sparse_retriever, alpha=0.5)
        docs, scores = retriever.retrieve("query", k=5)

        # Each doc appears in only one retriever
        # dense_score=1.0, sparse_score=0.0 → hybrid = 0.5*1.0 + 0.5*0.0 = 0.5
        # dense_score=0.0, sparse_score=1.0 → hybrid = 0.5*0.0 + 0.5*1.0 = 0.5
        assert len(docs) == 2
        assert len(scores) == 2
        assert all(score == 0.5 for score in scores)

    def test_overlapping_documents(self, mock_dense_retriever, mock_sparse_retriever):
        """Test handling of overlapping documents between retrievers."""
        # Same document returned by both retrievers
        shared_doc = Document(id=1, content="Shared doc")

        mock_dense_retriever.retrieve.return_value = ([shared_doc], [0.9])
        mock_sparse_retriever.retrieve.return_value = ([shared_doc], [0.7])

        retriever = HybridRetriever(mock_dense_retriever, mock_sparse_retriever, alpha=0.5)
        docs, scores = retriever.retrieve("query", k=5)

        # Should return document only once with fused score
        assert len(docs) == 1
        assert len(scores) == 1
        assert docs[0].id == 1

    def test_large_result_sets(self, mock_dense_retriever, mock_sparse_retriever):
        """Test with large result sets from both retrievers."""
        # Create large result sets
        dense_docs = [Document(id=i, content=f"Dense {i}") for i in range(100)]
        dense_scores = [0.9 - i * 0.001 for i in range(100)]

        sparse_docs = [Document(id=i + 50, content=f"Sparse {i}") for i in range(100)]
        sparse_scores = [0.8 - i * 0.001 for i in range(100)]

        mock_dense_retriever.retrieve.return_value = (dense_docs, dense_scores)
        mock_sparse_retriever.retrieve.return_value = (sparse_docs, sparse_scores)

        retriever = HybridRetriever(mock_dense_retriever, mock_sparse_retriever, alpha=0.5)
        docs, scores = retriever.retrieve("query", k=10)

        # Should return top k results
        assert len(docs) == 10
        assert len(scores) == 10

        # Scores should be in descending order
        assert scores == sorted(scores, reverse=True)

    def test_extreme_alpha_values(self, mock_dense_retriever, mock_sparse_retriever):
        """Test with extreme but valid alpha values."""
        # Test pure dense (alpha=0.0)
        retriever_dense = HybridRetriever(mock_dense_retriever, mock_sparse_retriever, alpha=0.0)
        docs, _ = retriever_dense.retrieve("query", k=5)

        # Should heavily favor dense results
        assert len(docs) >= 1

        # Test pure sparse (alpha=1.0)
        retriever_sparse = HybridRetriever(mock_dense_retriever, mock_sparse_retriever, alpha=1.0)
        docs, _ = retriever_sparse.retrieve("query", k=5)

        # Should heavily favor sparse results
        assert len(docs) >= 1

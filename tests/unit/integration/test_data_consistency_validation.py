"""
Tests for data consistency across storage layers.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.services.etl import ETLService
from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage
from local_rag_backend.infrastructure.retrieval.dense_faiss import DenseFaissRetriever
from local_rag_backend.infrastructure.retrieval.hybrid import HybridRetriever
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever

pytestmark = [pytest.mark.consistency]


class TestDataConsistencyValidation:
    """Test data consistency across all components."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def doc_storage(self, in_memory_sqlite):
        """Document storage with test database."""
        return SqlDocumentStorage(session_factory=in_memory_sqlite)

    @pytest.fixture
    def vector_storage(self, temp_dir):
        """Vector storage with temporary files."""
        return FaissVectorStorage(str(temp_dir / "test.faiss"), str(temp_dir / "test.pkl"), dim=4)

    @pytest.fixture
    def mock_embedder(self):
        """Mock embedder with consistent dimensions."""
        embedder = Mock()
        embedder.embed.side_effect = lambda texts: [
            [float(i), float(i + 1), float(i + 2), float(i + 3)] for i in range(len(texts))
        ]
        return embedder

    @pytest.fixture
    def etl_service(self, doc_storage, vector_storage, mock_embedder):
        """ETL service with real storage components."""
        return ETLService(doc_storage, vector_storage, mock_embedder)

    def test_document_vector_id_consistency(self, etl_service, doc_storage, vector_storage):
        """Test that document IDs and vector IDs remain consistent."""
        texts = ["Document 1", "Document 2", "Document 3"]

        # Ingest documents
        doc_ids = etl_service.ingest(texts)

        # Verify documents are stored with correct IDs
        stored_docs = doc_storage.get(doc_ids)
        assert len(stored_docs) == len(texts)
        assert [doc.id for doc in stored_docs] == doc_ids

        # Verify vector storage has same IDs
        assert len(vector_storage.faiss_index.id_map) == len(doc_ids)
        assert vector_storage.faiss_index.id_map == doc_ids

    def test_retrieval_document_correspondence(
        self, etl_service, doc_storage, vector_storage, mock_embedder
    ):
        """Test that retrieval results correspond to actual stored documents."""
        texts = ["Apple fruit", "Banana fruit", "Orange fruit"]
        doc_ids = etl_service.ingest(texts)

        # Create dense retriever
        retriever = DenseFaissRetriever(mock_embedder, vector_storage, doc_storage)

        # Retrieve documents
        docs, scores = retriever.retrieve("fruit", k=3)

        # Verify retrieved documents match stored documents
        assert len(docs) == len(scores)
        for doc, score in zip(docs, scores, strict=False):
            # Document should exist in storage
            stored_doc = doc_storage.get([doc.id])[0]
            assert stored_doc.id == doc.id
            assert stored_doc.content == doc.content

            # Score should be valid
            assert 0.0 <= score <= 1.0

    def test_score_document_order_preservation(
        self, etl_service, doc_storage, vector_storage, mock_embedder
    ):
        """Test that score-document order is preserved across retrievals."""
        texts = ["First document", "Second document", "Third document"]
        doc_ids = etl_service.ingest(texts)

        retriever = DenseFaissRetriever(mock_embedder, vector_storage, doc_storage)

        # Multiple retrievals should return consistent ordering
        for _ in range(5):
            docs, scores = retriever.retrieve("document", k=3)

            # Scores should be in descending order
            assert scores == sorted(scores, reverse=True)

            # Document order should correspond to score order
            for i in range(len(docs) - 1):
                assert scores[i] >= scores[i + 1]

    def test_hybrid_retriever_consistency(
        self, etl_service, doc_storage, vector_storage, mock_embedder
    ):
        """Test consistency in hybrid retriever score fusion."""
        texts = ["Machine learning algorithms", "Deep learning networks", "Neural network training"]
        doc_ids = etl_service.ingest(texts)

        # Get corpus for BM25
        from local_rag_backend.utils import get_corpus_and_ids

        corpus, corpus_ids = get_corpus_and_ids(doc_storage)

        # Create retrievers
        dense_retriever = DenseFaissRetriever(mock_embedder, vector_storage, doc_storage)
        sparse_retriever = SparseBM25Retriever(corpus, corpus_ids, doc_storage)
        hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever, alpha=0.5)

        # Test multiple alpha values
        for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
            hybrid = HybridRetriever(dense_retriever, sparse_retriever, alpha=alpha)
            docs, scores = hybrid.retrieve("learning", k=3)

            # Results should be consistent
            assert len(docs) == len(scores)
            assert all(isinstance(doc, Document) for doc in docs)
            assert all(isinstance(score, (int, float)) for score in scores)
            assert all(0.0 <= score <= 1.0 for score in scores)

    def test_missing_document_handling_consistency(
        self, doc_storage, vector_storage, mock_embedder
    ):
        """Test consistent handling when documents are missing from storage."""
        # Manually add to vector storage without document storage
        vector_storage.upsert([999], [[0.1, 0.2, 0.3, 0.4]])

        retriever = DenseFaissRetriever(mock_embedder, vector_storage, doc_storage)

        # Should handle missing documents gracefully
        docs, scores = retriever.retrieve("test", k=5)

        # Should return empty or only valid documents
        assert len(docs) == len(scores)
        for doc in docs:
            # Each returned document should exist in storage
            stored_docs = doc_storage.get([doc.id])
            assert len(stored_docs) == 1

    def test_concurrent_modification_consistency(
        self, etl_service, doc_storage, vector_storage, mock_embedder
    ):
        """Test consistency under simulated concurrent modifications."""
        # Initial ingestion
        initial_texts = ["Doc 1", "Doc 2", "Doc 3"]
        initial_ids = etl_service.ingest(initial_texts)

        retriever = DenseFaissRetriever(mock_embedder, vector_storage, doc_storage)

        # Get initial state
        initial_docs, _initial_scores = retriever.retrieve("Doc", k=5)
        initial_count = len(initial_docs)

        # Add more documents
        additional_texts = ["Doc 4", "Doc 5"]
        additional_ids = etl_service.ingest(additional_texts)

        # Verify consistency after addition
        final_docs, _final_scores = retriever.retrieve("Doc", k=10)

        # Should have more documents now
        assert len(final_docs) >= initial_count

        # All document IDs should be valid
        all_ids = initial_ids + additional_ids
        for doc in final_docs:
            assert doc.id in all_ids

    def test_embedding_dimension_consistency(self, doc_storage, vector_storage, mock_embedder):
        """Test that embedding dimensions remain consistent."""

        # Mock embedder to return consistent dimensions
        def consistent_embed(texts):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        mock_embedder.embed.side_effect = consistent_embed

        etl = ETLService(doc_storage, vector_storage, mock_embedder)

        # Multiple ingestions should maintain dimension consistency
        for i in range(3):
            texts = [f"Batch {i} doc {j}" for j in range(2)]
            doc_ids = etl.ingest(texts)
            assert len(doc_ids) == 2

        # Vector storage should have consistent dimensions
        assert len(vector_storage.faiss_index.id_map) == 6

        # All vectors should have same dimension
        retriever = DenseFaissRetriever(mock_embedder, vector_storage, doc_storage)
        docs, _scores = retriever.retrieve("doc", k=10)
        assert len(docs) <= 6

    def test_score_normalization_consistency(
        self, etl_service, doc_storage, vector_storage, mock_embedder
    ):
        """Test that score normalization is consistent across retrievals."""
        texts = ["Similar content", "Similar content", "Different content entirely"]
        doc_ids = etl_service.ingest(texts)

        retriever = DenseFaissRetriever(mock_embedder, vector_storage, doc_storage)

        # Multiple queries should have consistent score ranges
        queries = ["Similar", "content", "Different"]

        for query in queries:
            _docs, scores = retriever.retrieve(query, k=3)

            if scores:  # If any results returned
                # Scores should be normalized to [0, 1]
                assert all(0.0 <= score <= 1.0 for score in scores)

                # Scores should be in descending order
                assert scores == sorted(scores, reverse=True)

    def test_empty_corpus_consistency(self, doc_storage, vector_storage, mock_embedder):
        """Test consistent behavior with empty corpus."""
        # No documents ingested
        retriever = DenseFaissRetriever(mock_embedder, vector_storage, doc_storage)

        # Should handle empty corpus gracefully
        docs, scores = retriever.retrieve("any query", k=5)
        assert docs == []
        assert scores == []

        # Multiple calls should be consistent
        for _ in range(3):
            docs2, scores2 = retriever.retrieve("different query", k=10)
            assert docs2 == []
            assert scores2 == []

    def test_large_scale_consistency(self, etl_service, doc_storage, vector_storage, mock_embedder):
        """Test consistency with larger scale operations."""
        # Ingest larger batch
        texts = [
            f"Document number {i} with unique content about topic {i % 10}" for i in range(100)
        ]
        doc_ids = etl_service.ingest(texts)

        assert len(doc_ids) == 100
        assert len(set(doc_ids)) == 100  # All unique

        # Verify storage consistency
        stored_docs = doc_storage.get(doc_ids)
        assert len(stored_docs) == 100

        # Verify vector storage consistency
        assert len(vector_storage.faiss_index.id_map) == 100

        # Test retrieval consistency
        retriever = DenseFaissRetriever(mock_embedder, vector_storage, doc_storage)

        for k in [1, 5, 10, 50]:
            docs, scores = retriever.retrieve("content", k=k)
            assert len(docs) <= k
            assert len(docs) == len(scores)

            # All returned documents should be valid
            for doc in docs:
                assert doc.id in doc_ids
                assert doc.content in texts

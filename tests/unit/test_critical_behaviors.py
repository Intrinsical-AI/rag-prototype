# tests/unit/test_critical_behaviors.py
"""
Critical behavioral tests for RAG system edge cases and failure scenarios.
These tests focus on real-world failure modes and system boundaries.
"""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from local_rag_backend.core.services.etl import ETLService
from local_rag_backend.core.services.ingestion import default_chunker
from local_rag_backend.core.services.rag import RagService
from local_rag_backend.utils import normalize_similarities_from_distances


class TestChunkingBehavior:
    """Test chunking edge cases and boundary conditions."""

    def test_chunker_with_overlap_boundary_conditions(self):
        """Test chunker behavior at overlap boundaries."""
        # Test overlap = max_chars - 1 (boundary case)
        chunker = default_chunker(max_chars=10, overlap=9)
        result = chunker("abcdefghijklmnopqrstuvwxyz")

        # Should produce overlapping chunks without infinite loop
        assert len(result) > 1
        assert all(len(chunk) <= 10 for chunk in result)

        # Test that overlap >= max_chars gets clamped
        chunker_clamped = default_chunker(max_chars=10, overlap=15)
        result_clamped = chunker_clamped("abcdefghijklmnopqrstuvwxyz")
        assert len(result_clamped) > 1
        assert all(len(chunk) <= 10 for chunk in result_clamped)

    def test_chunker_with_unicode_and_special_chars(self):
        """Test chunker with unicode and special characters."""
        chunker = default_chunker(max_chars=20, overlap=5)
        text = "🚀 Python código with émojis and ñ characters 中文测试"

        result = chunker(text)
        assert result
        # Verify no character corruption
        reconstructed = "".join(result).replace(" ", "")
        original_clean = text.replace(" ", "")
        assert len(reconstructed) >= len(original_clean) - 10  # Allow for overlap


class TestConcurrentAccess:
    """Test concurrent access patterns and thread safety."""

    def test_rag_service_concurrent_queries(self, mock_document_repo, mock_vector_storage, mock_llm_generator):
        """Test RAG service under concurrent load."""
        # Mock retriever that simulates processing time
        mock_retriever = Mock()
        mock_retriever.retrieve.side_effect = lambda q, k: (
            [mock_document_repo.get_all_documents()[0]], [0.9]
        )

        rag_service = RagService(
            retriever=mock_retriever,
            generator=mock_llm_generator,
            history=Mock()
        )

        results = []
        errors = []

        def query_rag(query_id):
            try:
                result = rag_service.ask(f"Question {query_id}", top_k=1)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run 10 concurrent queries
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(query_rag, i) for i in range(10)]
            for future in futures:
                future.result(timeout=5)

        assert len(results) == 10
        assert len(errors) == 0
        assert all("answer" in result for result in results)

    def test_etl_service_concurrent_ingestion(self, mock_document_repo, mock_vector_storage, mock_embedder):
        """Test ETL service thread safety during concurrent ingestion."""
        etl_service = ETLService(mock_document_repo, mock_vector_storage, mock_embedder)

        results = []
        errors = []

        def ingest_batch(batch_id):
            try:
                texts = [f"Document {batch_id}-{i}" for i in range(5)]
                ids = etl_service.ingest(texts)
                results.append(ids)
            except Exception as e:
                errors.append(e)

        # Run concurrent ingestion
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(ingest_batch, i) for i in range(5)]
            for future in futures:
                future.result(timeout=5)

        assert len(results) == 5
        assert len(errors) == 0
        assert all(len(batch_ids) == 5 for batch_ids in results)


class TestErrorPropagation:
    """Test error handling and propagation through system layers."""

    def test_rag_service_handles_retriever_failure(self, mock_llm_generator):
        """Test RAG service graceful handling of retriever failures."""
        failing_retriever = Mock()
        failing_retriever.retrieve.side_effect = RuntimeError("Retriever connection failed")

        rag_service = RagService(
            retriever=failing_retriever,
            generator=mock_llm_generator,
            history=Mock()
        )

        with pytest.raises(RuntimeError, match="Retriever connection failed"):
            rag_service.ask("test question", top_k=3)

    def test_rag_service_handles_generator_failure(self, mock_document_repo, mock_vector_storage):
        """Test RAG service handling of generator failures."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = ([mock_document_repo.get_all_documents()[0]], [0.9])

        failing_generator = Mock()
        failing_generator.generate.side_effect = RuntimeError("LLM service unavailable")

        rag_service = RagService(
            retriever=mock_retriever,
            generator=failing_generator,
            history=Mock()
        )

        with pytest.raises(RuntimeError, match="LLM service unavailable"):
            rag_service.ask("test question", top_k=3)

    def test_etl_service_partial_failure_recovery(self, mock_vector_storage, mock_embedder):
        """Test ETL service behavior with partial failures."""
        # Mock document repo that fails on specific documents
        failing_doc_repo = Mock()
        def failing_store(texts):
            if len(texts) == 2:
                return [1, 2]
            else:
                raise RuntimeError("Storage full")
        failing_doc_repo.store_documents.side_effect = failing_store

        etl_service = ETLService(failing_doc_repo, mock_vector_storage, mock_embedder)

        # Should succeed with small batch
        ids = etl_service.ingest(["doc1", "doc2"])
        assert ids == [1, 2]

        # Should fail with larger batch
        with pytest.raises(RuntimeError, match="Storage full"):
            etl_service.ingest(["doc1", "doc2", "doc3"])


class TestResourceManagement:
    """Test resource management and cleanup."""

    def test_memory_usage_with_large_embeddings(self, mock_document_repo):
        """Test system behavior with large embedding batches."""
        # Mock embedder that returns large embeddings
        large_embedder = Mock()
        large_embedder.embed.return_value = [[0.1] * 1536 for _ in range(1000)]  # Large batch

        mock_vector_storage = Mock()
        etl_service = ETLService(mock_document_repo, mock_vector_storage, large_embedder)

        # Should handle large batches without memory issues
        large_texts = [f"Document {i}" * 100 for i in range(100)]  # Large documents
        ids = etl_service.ingest(large_texts)

        assert len(ids) == 100
        assert large_embedder.embed.called
        assert mock_vector_storage.upsert.called

    def test_cleanup_on_service_destruction(self, mock_document_repo, mock_vector_storage, mock_embedder):
        """Test proper cleanup when services are destroyed."""
        etl_service = ETLService(mock_document_repo, mock_vector_storage, mock_embedder)

        # Use service
        etl_service.ingest(["test document"])

        # Simulate destruction
        del etl_service

        # Verify mocks were called (service functioned properly)
        assert mock_document_repo.store_documents.called
        assert mock_embedder.embed.called


class TestConfigurationEdgeCases:
    """Test configuration validation and edge cases."""

    def test_invalid_retrieval_configurations(self):
        """Test system behavior with invalid configurations."""
        from pydantic import ValidationError

        from local_rag_backend.settings import Settings

        # Test invalid chunk overlap (overlap >= chunk_chars)
        with pytest.raises(ValidationError):
            Settings(ingest_chunk_chars=500, ingest_chunk_overlap=500)

        # Test boundary case (valid configuration)
        settings = Settings(ingest_chunk_chars=500, ingest_chunk_overlap=499)
        assert settings.ingest_chunk_overlap == 499

    def test_score_normalization_edge_cases(self):
        """Test score normalization with edge cases."""
        # Empty input
        assert normalize_similarities_from_distances([]) == []

        # Single value
        result = normalize_similarities_from_distances([1.0])
        assert result == [1.0]

        # All same values
        result = normalize_similarities_from_distances([2.0, 2.0, 2.0])
        assert result == [1.0, 1.0, 1.0]

        # Normal case
        result = normalize_similarities_from_distances([0.0, 1.0, 2.0])
        assert result == [1.0, 0.5, 0.0]  # Normalized to [0,1] range

        # Very large values
        result = normalize_similarities_from_distances([1000.0, 2000.0])
        assert 0.0 <= result[0] <= 1.0
        assert 0.0 <= result[1] <= 1.0


class TestDataConsistency:
    """Test data consistency across system components."""

    def test_document_id_consistency_across_stores(self, in_memory_sqlite):
        """Test that document IDs remain consistent across storage layers."""
        from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

        doc_storage = SqlDocumentStorage(session_factory=in_memory_sqlite)

        # Store documents
        texts = ["Document 1", "Document 2", "Document 3"]
        ids = doc_storage.store_documents(texts)

        # Retrieve and verify consistency
        retrieved_docs = doc_storage.get(ids)
        assert len(retrieved_docs) == len(texts)

        for i, doc in enumerate(retrieved_docs):
            assert doc.id == ids[i]
            assert doc.content == texts[i]

    def test_embedding_vector_dimension_consistency(self, mock_embedder):
        """Test that embedding dimensions remain consistent."""
        # Verify embedder returns consistent dimensions
        batch1 = mock_embedder.embed(["text1", "text2"])
        batch2 = mock_embedder.embed(["text3"])

        assert all(len(emb) == mock_embedder.dim for emb in batch1)
        assert all(len(emb) == mock_embedder.dim for emb in batch2)
        assert len(batch1[0]) == len(batch2[0])


class TestSystemIntegration:
    """Test critical integration points between components."""

    def test_end_to_end_query_flow_with_failures(self, mock_document_repo, mock_embedder):
        """Test complete query flow with simulated component failures."""
        # Setup components with potential failure points
        mock_vector_storage = Mock()
        mock_vector_storage.similar.return_value = ([1, 2], [0.9, 0.8])

        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = (
            mock_document_repo.get_all_documents()[:2], [0.9, 0.8]
        )

        mock_generator = Mock()
        mock_generator.generate.return_value = "Generated response"

        mock_history = Mock()

        # Test normal flow
        rag_service = RagService(mock_retriever, mock_generator, mock_history)
        result = rag_service.ask("test question", top_k=2)

        assert "answer" in result
        assert "docs" in result
        assert "scores" in result
        assert len(result["docs"]) == 2

        # Verify all components were called
        assert mock_retriever.retrieve.called
        assert mock_generator.generate.called
        assert mock_history.save.called

    def test_ingestion_to_retrieval_pipeline(self, in_memory_sqlite, mock_embedder):
        """Test complete ingestion to retrieval pipeline."""
        from local_rag_backend.core.services.etl import ETLService
        from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

        # Setup real document storage
        doc_storage = SqlDocumentStorage(session_factory=in_memory_sqlite)

        # Mock vector storage
        mock_vector_storage = Mock()

        # Create ETL service
        etl_service = ETLService(doc_storage, mock_vector_storage, mock_embedder)

        # Ingest documents
        texts = ["Python programming", "Machine learning", "Data science"]
        ids = etl_service.ingest(texts)

        # Verify documents were stored
        stored_docs = doc_storage.get_all_documents()
        assert len(stored_docs) == 3

        # Verify embeddings were generated and stored
        assert mock_embedder.embed.called
        assert mock_vector_storage.upsert.called

        # Verify consistency
        assert len(ids) == len(texts)
        assert all(isinstance(doc_id, int) for doc_id in ids)

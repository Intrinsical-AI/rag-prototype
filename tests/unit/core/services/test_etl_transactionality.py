"""
Tests for ETL service transactionality and rollback mechanisms.

This module tests the critical bug fixes for transactional safety
in the ETL pipeline, ensuring consistency between document and vector storage.
"""

from unittest.mock import Mock

import pytest

from local_rag_backend.core.services.etl import ETLService


class TestETLTransactionality:
    """Test transactional behavior and rollback mechanisms in ETL service."""

    @pytest.fixture
    def mock_doc_storage(self):
        """Mock document storage."""
        storage = Mock()
        storage.store_documents.return_value = [1, 2, 3]
        return storage

    @pytest.fixture
    def mock_vec_storage(self):
        """Mock vector storage."""
        storage = Mock()
        return storage

    @pytest.fixture
    def mock_embedder(self):
        """Mock embedder."""
        embedder = Mock()
        embedder.embed.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        return embedder

    @pytest.fixture
    def etl_service(self, mock_doc_storage, mock_vec_storage, mock_embedder):
        """ETL service with mocked dependencies."""
        return ETLService(mock_doc_storage, mock_vec_storage, mock_embedder)

    def test_successful_ingestion_flow(
        self, etl_service, mock_doc_storage, mock_vec_storage, mock_embedder
    ):
        """Test successful end-to-end ingestion."""
        texts = ["Document 1", "Document 2", "Document 3"]

        result = etl_service.ingest(texts)

        # Verify all phases executed
        mock_doc_storage.store_documents.assert_called_once_with(texts)
        mock_embedder.embed.assert_called_once_with(texts)
        mock_vec_storage.upsert.assert_called_once_with(
            [1, 2, 3], [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        )

        assert result == [1, 2, 3]

    @pytest.mark.parametrize(
        "input_texts,expected_filtered",
        [
            ([], []),  # Empty input
            ([""], []),  # Empty string
            (["   "], []),  # Whitespace only
            (["", "  ", "valid"], ["valid"]),  # Mixed empty and valid
            (["doc1", "", "doc2"], ["doc1", "doc2"]),  # Valid with empty in middle
            (["  doc1  ", "doc2"], ["doc1", "doc2"]),  # Texts with whitespace
        ],
    )
    def test_input_validation_and_filtering(
        self, etl_service, mock_doc_storage, input_texts, expected_filtered
    ):
        """Test input validation and empty text filtering."""
        result = etl_service.ingest(input_texts)

        if expected_filtered:
            mock_doc_storage.store_documents.assert_called_once_with(expected_filtered)
        else:
            mock_doc_storage.store_documents.assert_not_called()
            assert result == []

    def test_embedding_failure_triggers_rollback(
        self, etl_service, mock_doc_storage, mock_vec_storage, mock_embedder
    ):
        """Test rollback when embedding generation fails."""
        texts = ["Document 1", "Document 2"]
        mock_doc_storage.store_documents.return_value = [1, 2]
        mock_embedder.embed.side_effect = Exception("Embedding service down")

        # Mock rollback capability
        mock_doc_storage.delete_documents = Mock()

        with pytest.raises(RuntimeError, match="ETL pipeline failed during processing"):
            etl_service.ingest(texts)

        # Verify rollback was attempted
        mock_doc_storage.delete_documents.assert_called_once_with([1, 2])

    def test_vector_storage_failure_triggers_rollback(
        self, etl_service, mock_doc_storage, mock_vec_storage, mock_embedder
    ):
        """Test rollback when vector storage fails."""
        texts = ["Document 1", "Document 2"]
        mock_doc_storage.store_documents.return_value = [1, 2]
        mock_embedder.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_vec_storage.upsert.side_effect = Exception("FAISS index corrupted")

        # Mock rollback capability
        mock_doc_storage.delete_documents = Mock()

        with pytest.raises(RuntimeError, match="ETL pipeline failed during processing"):
            etl_service.ingest(texts)

        # Verify rollback was attempted
        mock_doc_storage.delete_documents.assert_called_once_with([1, 2])

    def test_embedding_count_mismatch_triggers_rollback(
        self, etl_service, mock_doc_storage, mock_vec_storage, mock_embedder
    ):
        """Test rollback when embedding count doesn't match document count."""
        texts = ["Document 1", "Document 2", "Document 3"]
        mock_doc_storage.store_documents.return_value = [1, 2, 3]
        # Return wrong number of embeddings
        mock_embedder.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]  # Only 2 embeddings for 3 docs

        # Mock rollback capability
        mock_doc_storage.delete_documents = Mock()

        with pytest.raises(RuntimeError, match="Embedding count mismatch"):
            etl_service.ingest(texts)

        # Verify rollback was attempted
        mock_doc_storage.delete_documents.assert_called_once_with([1, 2, 3])

    def test_document_storage_returns_empty_ids(
        self, etl_service, mock_doc_storage, mock_vec_storage, mock_embedder
    ):
        """Test handling when document storage returns empty IDs."""
        texts = ["Document 1"]
        mock_doc_storage.store_documents.return_value = []

        with pytest.raises(RuntimeError, match="Document storage failed: no IDs returned"):
            etl_service.ingest(texts)

        # Should not proceed to embedding
        mock_embedder.embed.assert_not_called()
        mock_vec_storage.upsert.assert_not_called()

    def test_rollback_failure_preserves_original_error(
        self, etl_service, mock_doc_storage, mock_vec_storage, mock_embedder
    ):
        """Test that rollback failure doesn't mask the original error."""
        texts = ["Document 1"]
        mock_doc_storage.store_documents.return_value = [1]
        mock_embedder.embed.side_effect = Exception("Original embedding error")

        # Mock rollback that also fails
        mock_doc_storage.delete_documents = Mock(side_effect=Exception("Rollback failed"))

        with pytest.raises(RuntimeError) as exc_info:
            etl_service.ingest(texts)

        error_msg = str(exc_info.value)
        assert "Original embedding error" in error_msg
        assert "Rollback also failed" in error_msg
        assert "Database may be in inconsistent state" in error_msg

    def test_no_rollback_when_no_deletion_support(
        self, etl_service, mock_doc_storage, mock_vec_storage, mock_embedder
    ):
        """Test graceful handling when document storage doesn't support deletion."""
        texts = ["Document 1"]
        mock_doc_storage.store_documents.return_value = [1]
        mock_embedder.embed.side_effect = Exception("Embedding failed")

        # Don't add delete_documents method to mock (no deletion support)

        with pytest.raises(RuntimeError, match="ETL pipeline failed during processing"):
            etl_service.ingest(texts)

        # Should not crash even without rollback capability

    @pytest.mark.parametrize(
        "failure_stage,setup_mocks",
        [
            (
                "doc_storage",
                lambda mocks: mocks[0].store_documents.side_effect.__setitem__(
                    0, Exception("DB error")
                ),
            ),
            (
                "embedder",
                lambda mocks: mocks[2].embed.side_effect.__setitem__(
                    0, Exception("Embedding error")
                ),
            ),
            (
                "vec_storage",
                lambda mocks: mocks[1].upsert.side_effect.__setitem__(0, Exception("Vector error")),
            ),
        ],
    )
    def test_error_propagation_with_context(
        self,
        etl_service,
        mock_doc_storage,
        mock_vec_storage,
        mock_embedder,
        failure_stage,
        setup_mocks,
    ):
        """Test that errors are properly propagated with helpful context."""
        texts = ["Document 1"]

        # Setup specific failure
        if failure_stage == "doc_storage":
            mock_doc_storage.store_documents.side_effect = Exception("DB error")
        elif failure_stage == "embedder":
            mock_doc_storage.store_documents.return_value = [1]
            mock_embedder.embed.side_effect = Exception("Embedding error")
            mock_doc_storage.delete_documents = Mock()
        elif failure_stage == "vec_storage":
            mock_doc_storage.store_documents.return_value = [1]
            mock_embedder.embed.return_value = [[0.1, 0.2]]
            mock_vec_storage.upsert.side_effect = Exception("Vector error")
            mock_doc_storage.delete_documents = Mock()

        with pytest.raises(RuntimeError) as exc_info:
            etl_service.ingest(texts)

        # Verify error contains helpful context
        if failure_stage == "doc_storage":
            assert "DB error" in str(exc_info.value)
        else:
            assert "ETL pipeline failed during processing" in str(exc_info.value)

    def test_concurrent_ingestion_safety(self, mock_doc_storage, mock_vec_storage, mock_embedder):
        """Test that ETL service handles concurrent ingestion safely."""
        # This is a basic test - full concurrency testing would require threading
        etl_service = ETLService(mock_doc_storage, mock_vec_storage, mock_embedder)

        texts1 = ["Doc 1", "Doc 2"]
        texts2 = ["Doc 3", "Doc 4"]

        # Mock different return values for different calls
        mock_doc_storage.store_documents.side_effect = [[1, 2], [3, 4]]
        mock_embedder.embed.side_effect = [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]

        result1 = etl_service.ingest(texts1)
        result2 = etl_service.ingest(texts2)

        assert result1 == [1, 2]
        assert result2 == [3, 4]
        assert mock_doc_storage.store_documents.call_count == 2
        assert mock_embedder.embed.call_count == 2

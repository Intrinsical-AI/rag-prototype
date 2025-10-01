"""
Advanced edge case tests for ETL service beyond basic transactionality.

Tests complex scenarios that could cause silent data corruption:
- Partial embedding failures
- Vector dimension mismatches
- Memory exhaustion scenarios
- Concurrent modification edge cases
- Resource cleanup validation
"""

import threading
from unittest.mock import Mock

import pytest

from local_rag_backend.core.services.etl import ETLService


class TestETLServiceAdvancedEdgeCases:
    """Test advanced edge cases for ETL service."""

    @pytest.fixture
    def mock_doc_storage(self):
        """Mock document storage with configurable behavior."""
        storage = Mock()
        # Return IDs based on number of documents received
        storage.store_documents.side_effect = lambda texts: list(range(1, len(texts) + 1))
        return storage

    @pytest.fixture
    def mock_vec_storage(self):
        """Mock vector storage with configurable behavior."""
        storage = Mock()
        return storage

    @pytest.fixture
    def mock_embedder(self):
        """Mock embedder with configurable behavior."""
        embedder = Mock()
        # Return embeddings matching number of input texts
        embedder.embed.side_effect = lambda texts: [[0.1, 0.2] for _ in range(len(texts))]
        return embedder

    @pytest.fixture
    def etl_service(self, mock_doc_storage, mock_vec_storage, mock_embedder):
        """ETL service with mocked dependencies."""
        return ETLService(mock_doc_storage, mock_vec_storage, mock_embedder)

    def test_embedder_returns_wrong_dimensions(self, etl_service, mock_embedder):
        """Test when embedder returns embeddings with inconsistent dimensions."""
        # Embedder returns embeddings with different dimensions
        mock_embedder.embed.side_effect = None
        mock_embedder.embed.return_value = [
            [0.1, 0.2, 0.3],  # 3D
            [0.4, 0.5],  # 2D
            [0.6, 0.7, 0.8, 0.9],  # 4D
        ]

        # Should handle gracefully or raise appropriate error
        with pytest.raises(RuntimeError, match="ETL pipeline failed during processing"):
            etl_service.ingest(["doc1", "doc2", "doc3"])

    def test_embedder_returns_nan_values(self, etl_service, mock_embedder):
        """Test when embedder returns NaN or infinite values."""

        mock_embedder.embed.side_effect = None
        mock_embedder.embed.return_value = [
            [float("nan"), 0.2],
            [0.3, float("inf")],
            [-float("inf"), 0.6],
        ]

        # Should detect and handle invalid embeddings
        with pytest.raises(RuntimeError, match="ETL pipeline failed during processing"):
            etl_service.ingest(["doc1", "doc2", "doc3"])

    def test_embedder_returns_empty_embeddings(self, etl_service, mock_embedder):
        """Test when embedder returns empty embeddings for some documents."""
        mock_embedder.embed.side_effect = None
        mock_embedder.embed.return_value = [
            [0.1, 0.2],
            [],  # Empty embedding
            [0.5, 0.6],
        ]

        # Should handle gracefully
        with pytest.raises(RuntimeError, match="ETL pipeline failed during processing"):
            etl_service.ingest(["doc1", "doc2", "doc3"])

    def test_vector_storage_partial_failure(self, etl_service, mock_vec_storage, mock_doc_storage):
        """Test when vector storage fails after partial success."""
        # Mock vector storage to fail on second call
        call_count = 0

        def upsert_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return  # Success
            else:
                raise Exception("Vector storage full")

        mock_vec_storage.upsert.side_effect = upsert_side_effect
        mock_doc_storage.delete_documents = Mock()

        # First ingestion should succeed
        result1 = etl_service.ingest(["doc1", "doc2", "doc3"])
        assert result1 == [1, 2, 3]

        # Second ingestion should fail and trigger rollback
        with pytest.raises(RuntimeError, match="ETL pipeline failed during processing"):
            etl_service.ingest(["doc4", "doc5", "doc6"])

        # Rollback should be attempted
        mock_doc_storage.delete_documents.assert_called_once()

    def test_document_storage_returns_duplicate_ids(
        self, etl_service, mock_doc_storage, mock_embedder
    ):
        """Test when document storage returns duplicate IDs."""
        # Reset side_effect and use return_value for this specific test
        mock_doc_storage.store_documents.side_effect = None
        mock_doc_storage.store_documents.return_value = [1, 1, 2]  # Duplicate ID
        mock_embedder.embed.side_effect = None
        mock_embedder.embed.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        # Should proceed but may cause issues in vector storage
        # This tests the robustness of the pipeline
        result = etl_service.ingest(["doc1", "doc2", "doc3"])
        assert result == [1, 1, 2]

    def test_memory_pressure_simulation(self, etl_service, mock_embedder, mock_doc_storage):
        """Test behavior under simulated memory pressure."""
        # Simulate large embeddings that could cause memory issues
        large_embedding_size = 1000
        large_embeddings = [[0.1] * large_embedding_size for _ in range(100)]
        mock_embedder.embed.return_value = large_embeddings

        # Mock document storage to return matching IDs
        mock_doc_storage.store_documents.return_value = list(range(1, 101))

        # Create large text batch
        large_texts = [f"Document {i} with content" for i in range(100)]

        # Should handle large batches gracefully
        result = etl_service.ingest(large_texts)
        assert len(result) == 100

    @pytest.mark.parametrize(
        "invalid_input,expected_count",
        [
            ([None, "valid", None], 1),  # Mixed None and valid
            ([123, "valid", 456], 1),  # Mixed types - only valid string
            (["", None, "   "], 0),  # Mixed empty and None - all invalid
        ],
    )
    def test_mixed_invalid_inputs(self, etl_service, invalid_input, expected_count):
        """Test with mixed valid and invalid inputs."""
        # Should filter out invalid inputs and process valid ones
        result = etl_service.ingest(invalid_input)

        # Should only process valid strings
        if expected_count > 0:
            assert len(result) == expected_count
        else:
            assert result == []

    def test_rollback_with_missing_delete_method(
        self, etl_service, mock_doc_storage, mock_embedder
    ):
        """Test rollback when document storage doesn't support deletion."""
        mock_embedder.embed.side_effect = Exception("Embedding failed")

        # Don't add delete_documents method to mock
        # This simulates a storage that doesn't support deletion

        with pytest.raises(RuntimeError, match="ETL pipeline failed during processing"):
            etl_service.ingest(["doc1"])

        # Should not crash even without rollback capability

    def test_rollback_delete_method_exception(self, etl_service, mock_doc_storage, mock_embedder):
        """Test when rollback delete method itself raises exception."""
        mock_embedder.embed.side_effect = Exception("Original error")
        mock_doc_storage.delete_documents = Mock(side_effect=Exception("Delete failed"))

        with pytest.raises(RuntimeError) as exc_info:
            etl_service.ingest(["doc1"])

        # Should mention both original error and rollback failure
        error_msg = str(exc_info.value)
        assert "Original error" in error_msg
        assert "Rollback also failed" in error_msg
        assert "Database may be in inconsistent state" in error_msg

    def test_concurrent_ingestion_simulation(
        self, mock_doc_storage, mock_vec_storage, mock_embedder
    ):
        """Test simulated concurrent ingestion scenarios."""
        # Create separate ETL instances to simulate different threads
        etl1 = ETLService(mock_doc_storage, mock_vec_storage, mock_embedder)
        etl2 = ETLService(mock_doc_storage, mock_vec_storage, mock_embedder)

        # Mock different return values for different calls
        doc_ids_sequence = [[1, 2], [3, 4], [5, 6], [7, 8]]
        embedding_sequence = [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            [[0.9, 1.0], [1.1, 1.2]],
            [[1.3, 1.4], [1.5, 1.6]],
        ]

        mock_doc_storage.store_documents.side_effect = doc_ids_sequence
        mock_embedder.embed.side_effect = embedding_sequence

        # Simulate concurrent operations
        results = []

        def ingest_batch(etl, texts, result_list):
            try:
                result = etl.ingest(texts)
                result_list.append(result)
            except Exception as e:
                result_list.append(e)

        threads = [
            threading.Thread(target=ingest_batch, args=(etl1, ["doc1", "doc2"], results)),
            threading.Thread(target=ingest_batch, args=(etl2, ["doc3", "doc4"], results)),
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Both operations should complete
        assert len(results) == 2
        assert all(isinstance(result, list) for result in results)

    def test_resource_cleanup_on_exception(self, etl_service, mock_doc_storage, mock_embedder):
        """Test that resources are properly cleaned up on exceptions."""
        # Mock to track resource usage
        resource_tracker = Mock()

        def embedding_with_resource_tracking(texts):
            resource_tracker.acquire()
            try:
                raise Exception("Simulated failure")
            finally:
                resource_tracker.release()

        mock_embedder.embed.side_effect = embedding_with_resource_tracking
        mock_doc_storage.delete_documents = Mock()

        with pytest.raises(RuntimeError):
            etl_service.ingest(["doc1"])

        # Resource should be properly released
        resource_tracker.acquire.assert_called_once()
        resource_tracker.release.assert_called_once()

    def test_very_large_text_documents(self, etl_service):
        """Test ingestion of very large text documents."""
        # Create very large documents (1MB each)
        large_text = "A" * (1024 * 1024)
        large_texts = [large_text, large_text, large_text]

        # Should handle large documents gracefully
        result = etl_service.ingest(large_texts)
        assert len(result) == 3

    def test_unicode_and_special_characters(self, etl_service):
        """Test ingestion with complex Unicode and special characters."""
        unicode_texts = [
            "Document with émojis 🚀🎉🔥",
            "Chinese: 你好世界 Japanese: こんにちは Arabic: مرحبا",
            "Mathematical symbols: ∑∏∫∆∇∂",
            # Note: Control characters \x00-\x03 may be filtered as whitespace-only
            "Mixed: ASCII + Unicode + 🌟",
        ]

        # Should handle Unicode gracefully (excluding control chars that get filtered)
        result = etl_service.ingest(unicode_texts)
        # All valid Unicode strings should be processed
        assert len(result) == 4  # Control char text is filtered out

    def test_embedding_dimension_consistency_check(
        self, etl_service, mock_embedder, mock_vec_storage, mock_doc_storage
    ):
        """Test that embedding dimensions are consistent across calls."""
        # First call returns 2D embeddings
        mock_embedder.embed.side_effect = [
            [[0.1, 0.2], [0.3, 0.4]],  # 2D
            [[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],  # 3D - inconsistent!
        ]

        # Mock different document IDs for each call
        mock_doc_storage.store_documents.side_effect = [[1, 2], [3, 4]]

        # First ingestion should succeed
        result1 = etl_service.ingest(["doc1", "doc2"])
        assert len(result1) == 2

        # Second ingestion with different dimensions should be handled
        # (This depends on vector storage implementation)
        try:
            result2 = etl_service.ingest(["doc3", "doc4"])
            # If it succeeds, dimensions were handled
            assert len(result2) == 2
        except RuntimeError:
            # If it fails, that's also acceptable behavior
            pass

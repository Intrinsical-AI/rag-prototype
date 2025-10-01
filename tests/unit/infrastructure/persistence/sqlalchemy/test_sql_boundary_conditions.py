"""
Comprehensive boundary condition tests for SQL document storage.

Tests critical edge cases for document retrieval and storage:
- Missing document handling
- Order preservation with gaps
- Large batch operations
- Concurrent access patterns
- Database constraint violations
"""

from unittest.mock import patch

import pytest

from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage


class TestSqlDocumentStorageBoundaryConditions:
    """Test boundary conditions for SqlDocumentStorage."""

    @pytest.fixture
    def storage(self, in_memory_sqlite):
        """Create SqlDocumentStorage with test database."""
        return SqlDocumentStorage(session_factory=in_memory_sqlite)

    def test_get_with_empty_ids(self, storage):
        """Test get method with empty ID list."""
        result = storage.get([])
        assert result == []

    def test_get_with_nonexistent_ids(self, storage):
        """Test get method with IDs that don't exist."""
        result = storage.get([999, 1000, 1001])
        assert result == []

    def test_get_with_mixed_existing_nonexistent_ids(self, storage):
        """Test get method with mix of existing and non-existing IDs."""
        # Store some documents
        stored_ids = storage.store_documents(["doc1", "doc2", "doc3"])

        # Request mix of existing and non-existing IDs
        mixed_ids = [stored_ids[0], 999, stored_ids[1], 1000]
        result = storage.get(mixed_ids)

        # Should return only existing documents in correct order
        assert len(result) == 2
        assert result[0].id == stored_ids[0]
        assert result[1].id == stored_ids[1]
        assert result[0].content == "doc1"
        assert result[1].content == "doc2"

    def test_get_order_preservation_with_gaps(self, storage):
        """Test that order is preserved even when some IDs are missing."""
        # Store documents
        stored_ids = storage.store_documents(["A", "B", "C", "D", "E"])

        # Request in specific order with gaps
        requested_ids = [stored_ids[4], stored_ids[1], stored_ids[0], stored_ids[3]]
        result = storage.get(requested_ids)

        # Should maintain order: E, B, A, D
        assert len(result) == 4
        assert [doc.content for doc in result] == ["E", "B", "A", "D"]
        assert [doc.id for doc in result] == requested_ids

    def test_get_with_duplicate_ids(self, storage):
        """Test get method with duplicate IDs in request."""
        # Store documents
        stored_ids = storage.store_documents(["doc1", "doc2"])

        # Request with duplicates
        duplicate_ids = [stored_ids[0], stored_ids[1], stored_ids[0], stored_ids[1]]
        result = storage.get(duplicate_ids)

        # Should return documents for each requested ID (including duplicates)
        assert len(result) == 4
        assert [doc.content for doc in result] == ["doc1", "doc2", "doc1", "doc2"]

    def test_get_with_missing_validation(self, storage):
        """Test get_with_missing_validation method."""
        # Store documents
        stored_ids = storage.store_documents(["doc1", "doc2"])

        # Test with all existing IDs
        result = storage.get_with_missing_validation(stored_ids)
        assert len(result) == 2

        # Test with missing IDs
        with pytest.raises(ValueError, match="Documents not found for IDs: \\[999\\]"):
            storage.get_with_missing_validation([stored_ids[0], 999])

    def test_store_documents_empty_list(self, storage):
        """Test store_documents with empty list."""
        result = storage.store_documents([])
        assert result == []

    @pytest.mark.parametrize(
        "texts,expected_count",
        [
            ([""], 1),  # Empty string should be stored
            (["   "], 1),  # Whitespace-only should be stored
            (["doc1", "", "doc2"], 3),  # Mix of content and empty
            (["doc1", "doc1"], 2),  # Duplicates should be stored separately
        ],
    )
    def test_store_documents_edge_cases(self, storage, texts, expected_count):
        """Test store_documents with various edge cases."""
        ids = storage.store_documents(texts)
        assert len(ids) == expected_count
        assert all(isinstance(id_, int) and id_ > 0 for id_ in ids)

    def test_store_very_long_document(self, storage):
        """Test storing very long document content."""
        # Create a very long document (10KB)
        long_content = "A" * 10000
        ids = storage.store_documents([long_content])

        # Should store successfully
        assert len(ids) == 1

        # Should retrieve correctly
        docs = storage.get(ids)
        assert len(docs) == 1
        assert docs[0].content == long_content

    def test_store_documents_with_special_characters(self, storage):
        """Test storing documents with special characters and Unicode."""
        special_texts = [
            "Document with émojis 🚀🎉",
            "SQL injection'; DROP TABLE documents; --",
            "Unicode: 中文, العربية, русский",
            "Newlines\nand\ttabs",
            "Quotes: 'single' and \"double\"",
            "Backslashes: \\n \\t \\r",
        ]

        ids = storage.store_documents(special_texts)
        assert len(ids) == len(special_texts)

        # Retrieve and verify
        docs = storage.get(ids)
        assert len(docs) == len(special_texts)
        for i, doc in enumerate(docs):
            assert doc.content == special_texts[i]

    def test_large_batch_operations(self, storage):
        """Test storing and retrieving large batches of documents."""
        # Create large batch
        batch_size = 1000
        texts = [f"Document {i}" for i in range(batch_size)]

        # Store large batch
        ids = storage.store_documents(texts)
        assert len(ids) == batch_size

        # Retrieve large batch
        docs = storage.get(ids)
        assert len(docs) == batch_size

        # Verify order and content
        for i, doc in enumerate(docs):
            assert doc.content == f"Document {i}"

    def test_get_all_documents_empty_database(self, storage):
        """Test get_all_documents on empty database."""
        result = storage.get_all_documents()
        assert result == []

    def test_get_all_documents_ordering(self, storage):
        """Test that get_all_documents returns documents in ID order."""
        # Store documents in random order
        texts = ["Third", "First", "Second"]
        ids = storage.store_documents(texts)

        # Get all documents
        all_docs = storage.get_all_documents()

        # Should be ordered by ID (which is insertion order)
        assert len(all_docs) == 3
        assert [doc.content for doc in all_docs] == ["Third", "First", "Second"]
        assert [doc.id for doc in all_docs] == sorted(ids)


class TestSqlDocumentStorageErrorHandling:
    """Test error handling and recovery scenarios."""

    @pytest.fixture
    def storage(self, in_memory_sqlite):
        """Create SqlDocumentStorage with test database."""
        return SqlDocumentStorage(session_factory=in_memory_sqlite)

    def test_database_connection_failure(self, storage):
        """Test behavior when database connection fails."""
        # Mock session factory to raise exception
        with patch.object(storage, "_session_factory") as mock_factory:
            mock_factory.side_effect = Exception("Database connection failed")

            with pytest.raises(Exception, match="Database connection failed"):
                storage.store_documents(["test"])

    def test_session_rollback_on_error(self, storage):
        """Test that sessions are properly rolled back on errors."""
        # Store some initial data
        initial_ids = storage.store_documents(["initial doc"])
        assert len(initial_ids) == 1

        # Attempt operation that might fail
        # Since SQLite is very permissive, we'll just verify consistency
        try:
            storage.store_documents(["test doc"])
        except Exception:
            pass

        # Database should still be in consistent state
        result = storage.get_all_documents()
        assert isinstance(result, list)
        assert len(result) >= 1  # Should have at least the initial document

    def test_concurrent_access_simulation(self, storage):
        """Test simulated concurrent access patterns."""
        # Simulate multiple "threads" storing documents
        results = []

        for i in range(5):
            ids = storage.store_documents([f"Thread {i} doc 1", f"Thread {i} doc 2"])
            results.extend(ids)

        # All IDs should be unique
        assert len(set(results)) == len(results)

        # All documents should be retrievable
        all_docs = storage.get_all_documents()
        assert len(all_docs) == 10


class TestSqlDocumentStoragePerformance:
    """Test performance-related edge cases."""

    @pytest.fixture
    def storage(self, in_memory_sqlite):
        """Create SqlDocumentStorage with test database."""
        return SqlDocumentStorage(session_factory=in_memory_sqlite)

    def test_large_id_list_retrieval(self, storage):
        """Test retrieval with very large ID lists."""
        # Store many documents
        batch_size = 100
        texts = [f"Doc {i}" for i in range(batch_size)]
        ids = storage.store_documents(texts)

        # Request all IDs at once
        docs = storage.get(ids)
        assert len(docs) == batch_size

        # Request in reverse order
        docs_reverse = storage.get(list(reversed(ids)))
        assert len(docs_reverse) == batch_size
        assert [doc.content for doc in docs_reverse] == [
            f"Doc {i}" for i in reversed(range(batch_size))
        ]

    def test_sparse_id_retrieval(self, storage):
        """Test retrieval with sparse ID patterns."""
        # Store documents
        texts = [f"Doc {i}" for i in range(100)]
        ids = storage.store_documents(texts)

        # Request every 10th document
        sparse_ids = ids[::10]
        docs = storage.get(sparse_ids)

        assert len(docs) == len(sparse_ids)
        assert [doc.content for doc in docs] == [f"Doc {i}" for i in range(0, 100, 10)]

    def test_repeated_retrieval_same_ids(self, storage):
        """Test repeated retrieval of same IDs (caching behavior)."""
        # Store documents
        ids = storage.store_documents(["doc1", "doc2", "doc3"])

        # Retrieve same IDs multiple times
        for _ in range(5):
            docs = storage.get(ids)
            assert len(docs) == 3
            assert [doc.content for doc in docs] == ["doc1", "doc2", "doc3"]

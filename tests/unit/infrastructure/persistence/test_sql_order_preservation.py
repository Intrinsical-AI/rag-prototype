"""
Tests for SQL document order preservation.

This module tests the critical bug fix for preserving ID order
in SQL document retrieval operations.
"""

import pytest

from local_rag_backend.core.domain.entities import Document as DomainDocument
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage


class TestSqlOrderPreservation:
    """Test order preservation in SQL document retrieval."""

    @pytest.fixture
    def storage(self, in_memory_sqlite):
        """SQL document storage with in-memory database."""
        return SqlDocumentStorage(session_factory=in_memory_sqlite)

    @pytest.fixture
    def sample_docs_data(self):
        """Sample document texts for testing."""
        return [
            "First document content",
            "Second document content",
            "Third document content",
            "Fourth document content",
            "Fifth document content",
        ]

    def test_order_preservation_basic(self, storage, sample_docs_data):
        """Test basic order preservation functionality."""
        # Store documents
        stored_ids = storage.store_documents(sample_docs_data)
        assert len(stored_ids) == 5

        # Test different order requests
        test_cases = [
            ([stored_ids[0], stored_ids[1], stored_ids[2]], "sequential"),
            ([stored_ids[2], stored_ids[0], stored_ids[1]], "non-sequential"),
            ([stored_ids[4], stored_ids[1], stored_ids[3]], "mixed order"),
        ]

        for requested_ids, case_name in test_cases:
            result = storage.get(requested_ids)
            result_ids = [doc.id for doc in result]

            # Verify order is preserved
            assert result_ids == requested_ids, f"Order not preserved for {case_name}"

            # Verify correct content
            for i, doc in enumerate(result):
                expected_index = stored_ids.index(requested_ids[i])
                assert doc.content == sample_docs_data[expected_index]

    def test_order_preservation_with_missing_docs(self, storage, sample_docs_data):
        """Test order preservation when some documents are missing."""
        # Store some documents
        stored_ids = storage.store_documents(sample_docs_data[:3])  # Only store first 3

        # Request in different order, including non-existent ID
        requested_ids = [stored_ids[2], 999, stored_ids[0], stored_ids[1]]  # 999 doesn't exist
        result = storage.get(requested_ids)
        result_ids = [doc.id for doc in result]

        # Should return only existing docs in requested order
        expected_ids = [stored_ids[2], stored_ids[0], stored_ids[1]]
        assert result_ids == expected_ids

    def test_empty_ids_list(self, storage):
        """Test handling of empty IDs list."""
        result = storage.get([])
        assert result == []

    def test_all_ids_missing(self, storage):
        """Test when none of the requested IDs exist."""
        result = storage.get([999, 998, 997])
        assert result == []

    def test_duplicate_ids_handling(self, storage, sample_docs_data):
        """Test handling of duplicate IDs in request."""
        # Store documents
        stored_ids = storage.store_documents(sample_docs_data[:3])

        # Request with duplicates
        requested_ids = [
            stored_ids[0],
            stored_ids[1],
            stored_ids[0],
            stored_ids[2],
            stored_ids[1],
            stored_ids[0],
        ]
        result = storage.get(requested_ids)
        result_ids = [doc.id for doc in result]

        # Should preserve duplicates in original order
        assert result_ids == requested_ids

        # Should have correct content for each
        assert len(result) == 6
        assert result[0].content == result[2].content == result[5].content  # All same ID
        assert result[1].content == result[4].content  # All same ID

    def test_content_integrity(self, storage):
        """Test that document content is preserved correctly."""
        special_contents = [
            "First content with special chars: àáâãäå",
            "Second content\nwith\nnewlines",
            "Third content with 'quotes' and \"double quotes\"",
        ]

        stored_ids = storage.store_documents(special_contents)
        requested_ids = [stored_ids[1], stored_ids[0], stored_ids[2]]  # Different order

        result = storage.get(requested_ids)

        # Verify content integrity and order
        assert result[0].id == stored_ids[1]
        assert result[0].content == special_contents[1]
        assert result[1].id == stored_ids[0]
        assert result[1].content == special_contents[0]
        assert result[2].id == stored_ids[2]
        assert result[2].content == special_contents[2]

    def test_type_safety(self, storage, sample_docs_data):
        """Test that returned objects are correct domain types."""
        stored_ids = storage.store_documents(sample_docs_data[:3])
        result = storage.get(stored_ids)

        # Verify all results are domain documents
        assert all(isinstance(doc, DomainDocument) for doc in result)

        # Verify they have the expected attributes
        for doc in result:
            assert hasattr(doc, "id")
            assert hasattr(doc, "content")
            assert isinstance(doc.id, int)
            assert isinstance(doc.content, str)

    def test_large_id_list_performance(self, storage):
        """Test performance with larger ID lists."""
        # Store 20 documents
        large_content = [f"Document content {i}" for i in range(20)]
        stored_ids = storage.store_documents(large_content)

        # Request in reverse order
        requested_ids = stored_ids[::-1]
        result = storage.get(requested_ids)
        result_ids = [doc.id for doc in result]

        # Should maintain reverse order
        assert result_ids == requested_ids
        assert len(result) == 20

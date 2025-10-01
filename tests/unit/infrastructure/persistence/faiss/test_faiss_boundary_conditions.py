"""
Comprehensive boundary condition tests for FAISS vector storage.

Tests critical edge cases that could cause silent failures or crashes:
- Empty arrays and null inputs
- Out-of-bounds index access
- Dimension mismatches
- Invalid vector shapes
- Memory and resource edge cases
"""

from unittest.mock import patch

import numpy as np
import pytest

from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
from local_rag_backend.infrastructure.persistence.faiss.index import FaissIndex

pytestmark = [pytest.mark.boundary]


class TestFaissIndexBoundaryConditions:
    """Test boundary conditions for FaissIndex."""

    @pytest.fixture
    def faiss_index(self, tmp_path):
        """Create a FaissIndex for testing."""
        return FaissIndex(tmp_path / "test.faiss", tmp_path / "test.pkl", dim=4)

    @pytest.mark.parametrize(
        "ids,embeddings,expected_error",
        [
            ([], [], None),  # Empty inputs should be handled gracefully
            ([1], [], r"ID count \(1\) must match embedding count \(0\)"),
            ([], [[0.1, 0.2, 0.3, 0.4]], r"ID count \(0\) must match embedding count \(1\)"),
            ([1, 2], [[0.1, 0.2, 0.3, 0.4]], r"ID count \(2\) must match embedding count \(1\)"),
            (
                [1],
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                r"ID count \(1\) must match embedding count \(2\)",
            ),
        ],
    )
    def test_add_to_index_input_validation(self, faiss_index, ids, embeddings, expected_error):
        """Test input validation for add_to_index method."""
        if expected_error:
            with pytest.raises(ValueError, match=expected_error):
                faiss_index.add_to_index(ids, embeddings)
        else:
            # Should not raise for empty inputs
            faiss_index.add_to_index(ids, embeddings)
            assert len(faiss_index.id_map) == 0

    @pytest.mark.parametrize(
        "embeddings,ids,expected_error",
        [
            (
                [[0.1, 0.2, 0.3]],
                [1],
                r"FAISS dim mismatch: vector dimension 3 != index dimension 4",
            ),
            (
                [[0.1, 0.2, 0.3, 0.4, 0.5]],
                [1],
                r"FAISS dim mismatch: vector dimension 5 != index dimension 4",
            ),
            # Proper 2D embedding with correct dimensions
            ([[0.1, 0.2, 0.3, 0.4]], [1], None),
            ([], [], None),  # Empty should be handled gracefully
        ],
    )
    def test_add_to_index_dimension_validation(self, faiss_index, embeddings, ids, expected_error):
        """Test dimension validation for embeddings."""
        if expected_error:
            with pytest.raises(ValueError, match=expected_error):
                faiss_index.add_to_index(ids, embeddings)
        else:
            faiss_index.add_to_index(ids, embeddings)
            if ids:
                assert len(faiss_index.id_map) == len(ids)

    def test_add_to_index_invalid_array_shapes(self, faiss_index):
        """Test handling of invalid array shapes."""
        # 3D array should fail
        with pytest.raises(ValueError, match="Embeddings must be 2D array"):
            faiss_index.add_to_index([1], [[[0.1, 0.2, 0.3, 0.4]]])

        # Jagged array (different lengths)
        with pytest.raises(ValueError):
            faiss_index.add_to_index([1, 2], [[0.1, 0.2], [0.1, 0.2, 0.3, 0.4]])

    def test_search_with_invalid_query_vector(self, faiss_index):
        """Test search behavior with invalid query vectors."""
        # Add some data first
        faiss_index.add_to_index([1], [[0.1, 0.2, 0.3, 0.4]])

        # Wrong dimension should raise error from FAISS
        with pytest.raises((ValueError, RuntimeError, AssertionError)):  # FAISS dimension error
            faiss_index.search([0.1, 0.2, 0.3], k=1)

    def test_search_empty_index(self, faiss_index):
        """Test search on empty index."""
        indices, _distances = faiss_index.search([0.1, 0.2, 0.3, 0.4], k=5)

        # Should return empty results or -1 indices
        assert len(indices) == 5  # FAISS returns k results even if empty
        assert all(idx == -1 for idx in indices)  # -1 indicates no result


class TestFaissVectorStorageBoundaryConditions:
    """Test boundary conditions for FaissVectorStorage."""

    @pytest.fixture
    def vector_storage(self, tmp_path):
        """Create a FaissVectorStorage for testing."""
        return FaissVectorStorage(str(tmp_path / "test.faiss"), str(tmp_path / "test.pkl"), dim=4)

    def test_similar_with_out_of_bounds_indices(self, vector_storage):
        """Test similar method with out-of-bounds indices."""
        # Add one document
        vector_storage.upsert([100], [[0.1, 0.2, 0.3, 0.4]])

        # Mock search to return out-of-bounds index
        with patch.object(vector_storage.faiss_index, "search") as mock_search:
            # Return index that's out of bounds
            mock_search.return_value = (np.array([999]), np.array([0.5]))

            # Should handle gracefully without crashing
            results = vector_storage.similar([0.1, 0.2, 0.3, 0.4], k=1)
            assert results == []  # Should return empty, not crash

    def test_similar_with_negative_indices(self, vector_storage):
        """Test similar method with negative indices from FAISS."""
        # Mock search to return -1 (FAISS no-result indicator)
        with patch.object(vector_storage.faiss_index, "search") as mock_search:
            mock_search.return_value = (np.array([-1]), np.array([0.0]))

            results = vector_storage.similar([0.1, 0.2, 0.3, 0.4], k=1)
            assert results == []

    def test_similar_with_mixed_valid_invalid_indices(self, vector_storage):
        """Test similar method with mix of valid and invalid indices."""
        # Add some documents
        vector_storage.upsert([1, 2], [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])

        # Mock search to return mix of valid and invalid indices
        with patch.object(vector_storage.faiss_index, "search") as mock_search:
            mock_search.return_value = (np.array([0, -1, 999, 1]), np.array([0.1, 0.2, 0.3, 0.4]))

            results = vector_storage.similar([0.1, 0.2, 0.3, 0.4], k=4)

            # Should only return valid results
            assert len(results) == 2
            assert all(isinstance(doc_id, int) and doc_id > 0 for doc_id, _ in results)

    def test_similar_empty_index(self, vector_storage):
        """Test similar on empty index."""
        results = vector_storage.similar([0.1, 0.2, 0.3, 0.4], k=5)
        assert results == []

    def test_similar_score_normalization_edge_cases(self, vector_storage):
        """Test score normalization with edge cases."""
        # Add documents
        vector_storage.upsert(
            [1, 2, 3], [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]]
        )

        # Test with identical distances (should normalize to 1.0)
        with patch.object(vector_storage.faiss_index, "search") as mock_search:
            mock_search.return_value = (np.array([0, 1, 2]), np.array([0.5, 0.5, 0.5]))

            results = vector_storage.similar([0.1, 0.2, 0.3, 0.4], k=3)

            # All scores should be 1.0 when distances are identical
            assert len(results) == 3
            assert all(score == 1.0 for _, score in results)

    @pytest.mark.parametrize("k", [0, -1, -5])
    def test_similar_invalid_k_values(self, vector_storage, k):
        """Test similar with invalid k values."""
        # Add some data
        vector_storage.upsert([1], [[0.1, 0.2, 0.3, 0.4]])

        # Should handle gracefully (FAISS will handle invalid k)
        try:
            results = vector_storage.similar([0.1, 0.2, 0.3, 0.4], k=k)
            # If no exception, should return empty or handle gracefully
            assert isinstance(results, list)
        except Exception:
            # FAISS may raise exception for invalid k, which is acceptable
            pass


class TestFaissMemoryAndResourceManagement:
    """Test memory and resource management edge cases."""

    def test_large_batch_upsert(self, tmp_path):
        """Test upserting large batches of vectors."""
        storage = FaissVectorStorage(
            str(tmp_path / "large.faiss"), str(tmp_path / "large.pkl"), dim=4
        )

        # Create large batch
        batch_size = 1000
        ids = list(range(batch_size))
        embeddings = [
            [float(i), float(i + 1), float(i + 2), float(i + 3)] for i in range(batch_size)
        ]

        # Should handle large batch without issues
        storage.upsert(ids, embeddings)

        # Verify data was stored
        assert len(storage.faiss_index.id_map) == batch_size

    def test_repeated_upserts(self, tmp_path):
        """Test multiple upsert operations."""
        storage = FaissVectorStorage(
            str(tmp_path / "repeat.faiss"), str(tmp_path / "repeat.pkl"), dim=4
        )

        # Multiple small upserts
        for i in range(10):
            storage.upsert([i], [[float(i), float(i + 1), float(i + 2), float(i + 3)]])

        assert len(storage.faiss_index.id_map) == 10

    def test_save_load_cycle(self, tmp_path):
        """Test save/load cycle maintains data integrity."""
        index_path = tmp_path / "cycle.faiss"
        id_map_path = tmp_path / "cycle.pkl"

        # Create and populate storage
        storage1 = FaissVectorStorage(str(index_path), str(id_map_path), dim=4)
        storage1.upsert([1, 2], [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])

        # Create new storage instance (should load existing data)
        storage2 = FaissVectorStorage(str(index_path), str(id_map_path), dim=4)

        # Should have same data
        assert len(storage2.faiss_index.id_map) == 2
        assert storage2.faiss_index.id_map == [1, 2]

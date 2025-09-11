"""
Test hybrid retriever alpha parameter validation.
"""

import pytest

from local_rag_backend.infrastructure.retrieval.hybrid import HybridRetriever


class MockRetriever:
    """Mock retriever for testing."""

    def retrieve(self, query: str, k: int):
        return [], []


@pytest.mark.parametrize("alpha", [-0.1, 1.1, -1.0, 2.0])
def test_hybrid_raises_on_invalid_alpha(alpha):
    """Test that HybridRetriever raises ValueError for alpha outside [0, 1] range."""
    dense = MockRetriever()
    sparse = MockRetriever()

    with pytest.raises(ValueError, match="alpha debe estar en"):
        HybridRetriever(dense=dense, sparse=sparse, alpha=alpha)


def test_hybrid_accepts_valid_alpha():
    """Test that HybridRetriever accepts valid alpha values."""
    dense = MockRetriever()
    sparse = MockRetriever()

    # Boundary values should work
    HybridRetriever(dense=dense, sparse=sparse, alpha=0.0)
    HybridRetriever(dense=dense, sparse=sparse, alpha=1.0)

    # Mid-range values should work
    HybridRetriever(dense=dense, sparse=sparse, alpha=0.5)
    HybridRetriever(dense=dense, sparse=sparse, alpha=0.3)
    HybridRetriever(dense=dense, sparse=sparse, alpha=0.7)

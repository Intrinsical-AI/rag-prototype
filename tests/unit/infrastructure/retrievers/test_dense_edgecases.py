# tests/unit/infrastructure/retrievers/test_dense_edgecases.py
import pytest

from local_rag_backend.core.domain.retrieval import RetrievalRequest


def test_dense_k_le_zero_raises():
    """RetrievalRequest rejects top_k=0 at construction time."""
    with pytest.raises(ValueError, match="top_k must be positive"):
        RetrievalRequest(query="q", top_k=0, mode="dense")

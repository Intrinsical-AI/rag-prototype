# tests/unit/adapters/retrieval/test_hybrid_retriever.py
import pytest
from unittest.mock import Mock
from src.adapters.retrieval.hybrid import HybridRetriever
from src.core.ports import RetrieverPort # Asegúrate que este path sea correcto

@pytest.fixture
def mock_dense_retriever() -> Mock:
    retriever = Mock(spec=RetrieverPort)
    retriever.retrieve.return_value = ([], []) # Default: no devuelve nada
    return retriever

@pytest.fixture
def mock_sparse_retriever() -> Mock:
    retriever = Mock(spec=RetrieverPort)
    retriever.retrieve.return_value = ([], []) # Default: no devuelve nada
    return retriever

def test_hybrid_dense_only(mock_dense_retriever: Mock, mock_sparse_retriever: Mock):
    # Arrange
    dense_ids, dense_scores = [1, 2], [0.8, 0.7]
    mock_dense_retriever.retrieve.return_value = (dense_ids, dense_scores)
    mock_sparse_retriever.retrieve.return_value = ([], [])
    
    alpha = 0.5 # Weight for sparse, so (1-alpha) for dense
    hybrid_retriever = HybridRetriever(dense=mock_dense_retriever, sparse=mock_sparse_retriever, alpha=alpha)
    
    # Act
    retrieved_ids, retrieved_scores = hybrid_retriever.retrieve("test query")
    
    # Assert
    mock_dense_retriever.retrieve.assert_called_once_with("test query", 5) # k=5 es el default
    mock_sparse_retriever.retrieve.assert_called_once_with("test query", 5)

    assert retrieved_ids == [1, 2]
    # Scores should be (1-alpha) * original dense scores
    expected_scores = [s * (1 - alpha) for s in dense_scores]
    assertpytest.approx(retrieved_scores, rel=1e-9) == expected_scores

def test_hybrid_sparse_only(mock_dense_retriever: Mock, mock_sparse_retriever: Mock):
    # Arrange
    sparse_ids, sparse_scores = [3, 4], [0.9, 0.6]
    mock_dense_retriever.retrieve.return_value = ([], [])
    mock_sparse_retriever.retrieve.return_value = (sparse_ids, sparse_scores)

    alpha = 0.5
    hybrid_retriever = HybridRetriever(dense=mock_dense_retriever, sparse=mock_sparse_retriever, alpha=alpha)

    # Act
    retrieved_ids, retrieved_scores = hybrid_retriever.retrieve("test query")

    # Assert
    assert retrieved_ids == [3, 4]
    # Scores should be alpha * original sparse scores
    expected_scores = [s * alpha for s in sparse_scores]
    assert pytest.approx(retrieved_scores, rel=1e-9) == expected_scores

def test_hybrid_both_no_overlap(mock_dense_retriever: Mock, mock_sparse_retriever: Mock):
    # Arrange
    mock_dense_retriever.retrieve.return_value = ([1], [0.8])
    mock_sparse_retriever.retrieve.return_value = ([2], [0.9])
    
    alpha = 0.5
    hybrid_retriever = HybridRetriever(dense=mock_dense_retriever, sparse=mock_sparse_retriever, alpha=alpha)

    # Act
    retrieved_ids, retrieved_scores = hybrid_retriever.retrieve("test query")

    # Assert
    # Dense score: 0.8 * (1-0.5) = 0.4
    # Sparse score: 0.9 * 0.5 = 0.45
    # Expected order: doc 2 (sparse) then doc 1 (dense)
    assert retrieved_ids == [2, 1]
    assert pytest.approx(retrieved_scores, rel=1e-9) == [0.45, 0.4]

def test_hybrid_both_with_overlap(mock_dense_retriever: Mock, mock_sparse_retriever: Mock):
    # Arrange
    mock_dense_retriever.retrieve.return_value = ([1, 2], [0.8, 0.6]) # doc 1, doc 2
    mock_sparse_retriever.retrieve.return_value = ([2, 3], [0.9, 0.7]) # doc 2, doc 3
    
    alpha = 0.5
    hybrid_retriever = HybridRetriever(dense=mock_dense_retriever, sparse=mock_sparse_retriever, alpha=alpha)

    # Act
    retrieved_ids, retrieved_scores = hybrid_retriever.retrieve("test query")

    # Assert
    # Doc 1 (dense only): 0.8 * (1-0.5) = 0.4
    # Doc 2 (both): (0.6 * (1-0.5)) + (0.9 * 0.5) = 0.3 + 0.45 = 0.75
    # Doc 3 (sparse only): 0.7 * 0.5 = 0.35
    # Expected order: Doc 2, Doc 1, Doc 3
    assert retrieved_ids == [2, 1, 3]
    assert pytest.approx(retrieved_scores, rel=1e-9) == [0.75, 0.4, 0.35]

@pytest.mark.parametrize("alpha_val, expected_id_order, expected_scores_approx", [
    (0.0, [10], [0.8]),   # Dense only
    (1.0, [10], [0.7]),   # Sparse only
    (0.3, [10], [0.8*0.7 + 0.7*0.3]), # Weighted, 0.56 + 0.21 = 0.77
    (0.7, [10], [0.8*0.3 + 0.7*0.7]), # Weighted, 0.24 + 0.49 = 0.73
])
def test_hybrid_alpha_variations(
    mock_dense_retriever: Mock,
    mock_sparse_retriever: Mock,
    alpha_val: float,
    expected_id_order: list[int],
    expected_scores_approx: list[float]
):
    # Arrange
    # Use a single common document to clearly see alpha's effect
    mock_dense_retriever.retrieve.return_value = ([10], [0.8])
    mock_sparse_retriever.retrieve.return_value = ([10], [0.7])
    
    hybrid_retriever = HybridRetriever(
        dense=mock_dense_retriever, 
        sparse=mock_sparse_retriever, 
        alpha=alpha_val
    )

    # Act
    retrieved_ids, retrieved_scores = hybrid_retriever.retrieve("test query")

    # Assert
    assert retrieved_ids == expected_id_order
    assert pytest.approx(retrieved_scores, rel=1e-9) == expected_scores_approx

def test_hybrid_respects_k_param(mock_dense_retriever: Mock, mock_sparse_retriever: Mock):
    # Arrange
    # Both retrievers return more docs than k
    mock_dense_retriever.retrieve.return_value = ([1, 2, 3], [0.9, 0.8, 0.7])
    mock_sparse_retriever.retrieve.return_value = ([3, 4, 5], [0.85, 0.75, 0.65])
    
    alpha = 0.5
    k_val = 2 # We want only top 2 results
    hybrid_retriever = HybridRetriever(dense=mock_dense_retriever, sparse=mock_sparse_retriever, alpha=alpha)

    # Act
    retrieved_ids, retrieved_scores = hybrid_retriever.retrieve("test query", k=k_val)

    # Assert
    mock_dense_retriever.retrieve.assert_called_once_with("test query", k_val)
    mock_sparse_retriever.retrieve.assert_called_once_with("test query", k_val)
    
    assert len(retrieved_ids) == k_val
    assert len(retrieved_scores) == k_val
    # Scores for context:
    # Doc 1 (dense): 0.9 * 0.5 = 0.45
    # Doc 2 (dense): 0.8 * 0.5 = 0.4
    # Doc 3 (both): (0.7 * 0.5) + (0.85 * 0.5) = 0.35 + 0.425 = 0.775
    # Doc 4 (sparse): 0.75 * 0.5 = 0.375
    # Doc 5 (sparse): 0.65 * 0.5 = 0.325
    # Top 2 expected: Doc 3, Doc 1
    assert retrieved_ids == [3, 1]
    assert pytest.approx(retrieved_scores, rel=1e-9) == [0.775, 0.45]

def test_hybrid_empty_results_from_both(mock_dense_retriever: Mock, mock_sparse_retriever: Mock):
    # Arrange
    mock_dense_retriever.retrieve.return_value = ([], [])
    mock_sparse_retriever.retrieve.return_value = ([], [])
    
    hybrid_retriever = HybridRetriever(dense=mock_dense_retriever, sparse=mock_sparse_retriever, alpha=0.5)

    # Act
    retrieved_ids, retrieved_scores = hybrid_retriever.retrieve("test query")

    # Assert
    assert retrieved_ids == []
    assert retrieved_scores == []
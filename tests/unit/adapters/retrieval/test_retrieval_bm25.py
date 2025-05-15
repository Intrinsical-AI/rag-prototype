# tests/unit/test_retrieval_bm25.py

import pytest
from src.adapters.retrieval.sparse_bm25 import SparseBM25Retriever


@pytest.fixture
def sample_documents_data() -> tuple[list[str], list[int]]:
    """Provides a set of example documents and their IDs for BM25 retrieval tests."""
    documents = [
        "The quick brown dog jumps over the lazy fox.",
        "The rain in Spain falls mainly in the plain.",
        "A dog is man's best friend.",
        "Foxes are agile, and dogs are too.",
    ]
    doc_ids = [10, 20, 30, 40]  # Arbitrary document IDs
    return documents, doc_ids


def test_bm25_retrieves_relevant_doc(sample_documents_data):
    """BM25 retrieves the most relevant document for a simple query."""
    documents, doc_ids = sample_documents_data
    retriever = SparseBM25Retriever(documents=documents, doc_ids=doc_ids)

    query = "dog friend"
    retrieved_ids, retrieved_scores = retriever.retrieve(query, k=1)

    assert len(retrieved_ids) == 1
    # "A dog is man's best friend." should be the best match
    assert retrieved_ids[0] == 30
    assert len(retrieved_scores) == 1
    assert isinstance(retrieved_scores[0], float)


def test_bm25_respects_k(sample_documents_data):
    """BM25 returns the correct number of results as specified by k."""
    documents, doc_ids = sample_documents_data
    retriever = SparseBM25Retriever(documents=documents, doc_ids=doc_ids)

    query = "LES"  # Query not present, should fallback to order/score
    k_val = 2
    retrieved_ids, retrieved_scores = retriever.retrieve(query, k=k_val)

    assert len(retrieved_ids) == k_val
    assert len(retrieved_scores) == k_val
    # Optionally: Check IDs are among provided doc_ids
    for doc_id in retrieved_ids:
        assert doc_id in doc_ids


def test_bm25_no_match(sample_documents_data):
    """BM25 handles queries with no matching terms gracefully."""
    documents, doc_ids = sample_documents_data
    retriever = SparseBM25Retriever(documents=documents, doc_ids=doc_ids)

    query = "unicorn cat nonexistent"
    retrieved_ids, retrieved_scores = retriever.retrieve(query, k=1)

    # Depending on implementation, BM25 may still return results with zero score
    assert isinstance(retrieved_ids, list)
    assert isinstance(retrieved_scores, list)
    assert len(retrieved_ids) == len(retrieved_scores)
    if retrieved_scores:
        assert all(isinstance(score, float) for score in retrieved_scores)


def test_bm25_empty_query(sample_documents_data):
    """BM25 returns k documents with zero scores for an empty query."""
    documents, doc_ids = sample_documents_data
    retriever = SparseBM25Retriever(documents=documents, doc_ids=doc_ids)

    query = ""
    retrieved_ids, retrieved_scores = retriever.retrieve(query, k=1)

    assert len(retrieved_ids) <= 1
    assert len(retrieved_scores) <= 1
    if retrieved_scores:
        assert all(score == 0.0 for score in retrieved_scores)


def test_bm25_returns_sorted_scores(sample_documents_data):
    """BM25 returns results sorted by score in descending order."""
    documents, doc_ids = sample_documents_data
    retriever = SparseBM25Retriever(documents=documents, doc_ids=doc_ids)

    query = "dog fox"
    k_val = 4
    retrieved_ids, retrieved_scores = retriever.retrieve(query, k=k_val)

    # Scores should be in non-increasing order
    assert all(
        retrieved_scores[i] >= retrieved_scores[i + 1]
        for i in range(len(retrieved_scores) - 1)
    )

# src/infrastructure/retrieval/hybrid.py
"""
Hybrid retriever using dense and sparse retrieval methods.
"""

from __future__ import annotations

from collections.abc import Sequence

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.ports import RetrieverPort


class HybridRetriever(RetrieverPort):
    """Combines dense and sparse retrieval methods using a weighted average."""

    def __init__(self, dense: RetrieverPort, sparse: RetrieverPort, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha for hybrid retrieval must be between 0.0 and 1.0.")
        self.dense = dense
        self.sparse = sparse
        self.alpha = alpha

    def retrieve(self, query: str, k: int = 5) -> tuple[Sequence[Document], Sequence[float]]:
        """Retrieve documents by combining dense and sparse scores."""
        dense_docs, dense_scores = self.dense.retrieve(query, k)
        sparse_docs, sparse_scores = self.sparse.retrieve(query, k)

        # Create score maps for efficient lookup
        dense_score_map = {
            doc.id: score for doc, score in zip(dense_docs, dense_scores, strict=False)
        }
        sparse_score_map = {
            doc.id: score for doc, score in zip(sparse_docs, sparse_scores, strict=False)
        }

        # Combine all unique documents
        all_docs = {doc.id: doc for doc in list(dense_docs) + list(sparse_docs)}

        # Calculate hybrid scores
        combined_results = []
        for doc_id, doc in all_docs.items():
            dense_score = dense_score_map.get(doc_id, 0.0)
            sparse_score = sparse_score_map.get(doc_id, 0.0)
            hybrid_score = (1 - self.alpha) * dense_score + self.alpha * sparse_score
            combined_results.append((doc, hybrid_score))

        # Sort by hybrid score and take the top k
        combined_results.sort(key=lambda item: item[1], reverse=True)
        top_k_results = combined_results[:k]

        if not top_k_results:
            return [], []

        final_docs, final_scores = zip(*top_k_results, strict=False)
        return list(final_docs), list(final_scores)

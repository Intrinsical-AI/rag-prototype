"""
RAG Prototype - Intrinsical-AI (c) 2025
Author: Pablo Pintor
License: MIT

Module: Hybrid Retriever
Purpose: Combines dense and sparse retrieval methods using weighted score fusion.
         Provides best-of-both-worlds approach for document retrieval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_backend.core.ports import RetrieverPort

if TYPE_CHECKING:
    from collections.abc import Sequence

    from local_rag_backend.core.domain.entities import Document


class HybridRetriever(RetrieverPort):
    """Combines dense and sparse retrieval using weighted score fusion.

    This retriever leverages both semantic similarity (dense) and keyword matching
    (sparse) to provide comprehensive document retrieval. The alpha parameter
    controls the balance between the two approaches.

    Formula: hybrid_score = (1 - alpha) * dense_score + alpha * sparse_score
    - alpha = 0.0: Pure dense retrieval
    - alpha = 1.0: Pure sparse retrieval
    - alpha = 0.5: Equal weighting
    """

    def __init__(self, dense: RetrieverPort, sparse: RetrieverPort, alpha: float = 0.5):
        """Initialize hybrid retriever with dense and sparse components.

        Args:
            dense: Dense retriever for semantic similarity
            sparse: Sparse retriever for keyword matching
            alpha: Weight for sparse scores (0.0 to 1.0)

        Raises:
            ValueError: If alpha is not in [0.0, 1.0] range
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha for hybrid retrieval must be between 0.0 and 1.0.")
        self.dense = dense
        self.sparse = sparse
        self.alpha = alpha

    def retrieve(self, query: str, k: int = 5) -> tuple[Sequence[Document], Sequence[float]]:
        """Retrieve documents using hybrid dense-sparse fusion.

        Args:
            query: Search query text
            k: Maximum number of documents to return

        Returns:
            Tuple of (documents, hybrid_scores) ordered by relevance
        """
        # --- Parallel Retrieval ---
        dense_docs, dense_scores = self.dense.retrieve(query, k)
        sparse_docs, sparse_scores = self.sparse.retrieve(query, k)

        # --- Score Mapping ---
        dense_scores_by_id = {
            doc.id: score for doc, score in zip(dense_docs, dense_scores, strict=False)
        }
        sparse_scores_by_id = {
            doc.id: score for doc, score in zip(sparse_docs, sparse_scores, strict=False)
        }

        # --- Document Union ---
        all_docs_by_id = {doc.id: doc for doc in list(dense_docs) + list(sparse_docs)}

        # --- Score Fusion ---
        fused_results = []
        for doc_id, doc in all_docs_by_id.items():
            dense_score = dense_scores_by_id.get(doc_id, 0.0)
            sparse_score = sparse_scores_by_id.get(doc_id, 0.0)
            hybrid_score = (1 - self.alpha) * dense_score + self.alpha * sparse_score
            fused_results.append((doc, hybrid_score))

        # --- Ranking and Selection ---
        fused_results.sort(key=lambda item: item[1], reverse=True)
        top_results = fused_results[:k]

        if not top_results:
            return [], []

        docs, scores = zip(*top_results, strict=False)
        return list(docs), list(scores)

# src/infrastructure/retrieval/hybrid.py
"""
Hybrid retriever using dense and sparse retrieval methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from local_rag_backend.core.domain.retrieval import (
    RetrievalRequest,
    RetrievalResult,
    retrieval_result_from_pairs,
)
from local_rag_backend.core.ports import RetrieverPort

if TYPE_CHECKING:
    from collections.abc import Sequence

    from local_rag_backend.core.domain.entities import Document


class HybridRetriever(RetrieverPort):
    """Combines dense and sparse retrieval methods using a weighted average."""

    def __init__(self, dense: RetrieverPort, sparse: RetrieverPort, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha for hybrid retrieval must be between 0.0 and 1.0.")
        self.dense = dense
        self.sparse = sparse
        self.alpha = alpha

    @overload
    def retrieve(self, query: RetrievalRequest, k: int = 5) -> RetrievalResult: ...

    @overload
    def retrieve(self, query: str, k: int = 5) -> tuple[Sequence[Document], Sequence[float]]: ...

    def retrieve(
        self, query: str | RetrievalRequest, k: int = 5
    ) -> tuple[Sequence[Document], Sequence[float]] | RetrievalResult:
        """Retrieve documents by combining dense and sparse scores."""
        if isinstance(query, RetrievalRequest):
            request = query
            raw_dense = self.dense.retrieve(request)
            raw_sparse = self.sparse.retrieve(request)
            dense_docs: list[Document]
            dense_scores: list[float]
            if isinstance(raw_dense, RetrievalResult):
                dense_docs = list(raw_dense.documents)
                dense_scores = list(raw_dense.scores)
            else:
                legacy_dense_docs, legacy_dense_scores = raw_dense
                dense_docs = list(legacy_dense_docs)
                dense_scores = list(legacy_dense_scores)
            sparse_docs: list[Document]
            sparse_scores: list[float]
            if isinstance(raw_sparse, RetrievalResult):
                sparse_docs = list(raw_sparse.documents)
                sparse_scores = list(raw_sparse.scores)
            else:
                legacy_sparse_docs, legacy_sparse_scores = raw_sparse
                sparse_docs = list(legacy_sparse_docs)
                sparse_scores = list(legacy_sparse_scores)
        else:
            request = None
            dense_legacy = self.dense.retrieve(query, k)
            sparse_legacy = self.sparse.retrieve(query, k)
            legacy_dense_docs, legacy_dense_scores = dense_legacy
            legacy_sparse_docs, legacy_sparse_scores = sparse_legacy
            dense_docs = list(legacy_dense_docs)
            dense_scores = list(legacy_dense_scores)
            sparse_docs = list(legacy_sparse_docs)
            sparse_scores = list(legacy_sparse_scores)

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
            if request is not None:
                return RetrievalResult(items=(), mode_used="hybrid", backend_used="legacy_hybrid")
            return [], []

        final_docs, final_scores = zip(*top_k_results, strict=False)
        if request is not None:
            return retrieval_result_from_pairs(
                docs=list(final_docs),
                scores=list(final_scores),
                mode_used="hybrid",
                backend_used="legacy_hybrid",
                stage="hybrid",
            )
        return list(final_docs), list(final_scores)

# src/infrastructure/retrieval/hybrid.py
"""
Hybrid retriever using dense and sparse retrieval methods.
"""

from __future__ import annotations

from local_rag_backend.core.domain.retrieval import (
    RetrievalRequest,
    RetrievalResult,
    retrieval_result_from_pairs,
)
from local_rag_backend.core.ports import RetrieverPort


def combine_hybrid_results(
    *,
    dense_result: RetrievalResult,
    sparse_result: RetrievalResult,
    alpha: float,
    top_k: int,
) -> RetrievalResult:
    """Combine dense and sparse results with the same semantics as HybridRetriever."""
    dense_docs = list(dense_result.documents)
    dense_scores = list(dense_result.scores)
    sparse_docs = list(sparse_result.documents)
    sparse_scores = list(sparse_result.scores)

    dense_score_map = {doc.id: score for doc, score in zip(dense_docs, dense_scores, strict=False)}
    sparse_score_map = {
        doc.id: score for doc, score in zip(sparse_docs, sparse_scores, strict=False)
    }

    all_docs = {doc.id: doc for doc in list(dense_docs) + list(sparse_docs)}

    combined_results = []
    for doc_id, doc in all_docs.items():
        dense_score = dense_score_map.get(doc_id, 0.0)
        sparse_score = sparse_score_map.get(doc_id, 0.0)
        hybrid_score = (1 - alpha) * dense_score + alpha * sparse_score
        combined_results.append((doc, hybrid_score))

    combined_results.sort(key=lambda item: item[1], reverse=True)
    top_k_results = combined_results[:top_k]

    if not top_k_results:
        return RetrievalResult(items=(), mode_used="hybrid", backend_used="legacy_hybrid")

    final_docs, final_scores = zip(*top_k_results, strict=False)
    return retrieval_result_from_pairs(
        docs=list(final_docs),
        scores=list(final_scores),
        mode_used="hybrid",
        backend_used="legacy_hybrid",
        stage="hybrid",
    )


class HybridRetriever(RetrieverPort):
    """Combines dense and sparse retrieval methods using a weighted average."""

    def __init__(self, dense: RetrieverPort, sparse: RetrieverPort, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha for hybrid retrieval must be between 0.0 and 1.0.")
        self.dense = dense
        self.sparse = sparse
        self.alpha = alpha

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Retrieve documents by combining dense and sparse scores."""
        raw_dense = self.dense.retrieve(request)
        raw_sparse = self.sparse.retrieve(request)
        return combine_hybrid_results(
            dense_result=raw_dense,
            sparse_result=raw_sparse,
            alpha=self.alpha,
            top_k=request.top_k,
        )


# === file: src/adapters/retrieval/hybrid.py ===
"""Híbrido denso + BM25 con suma ponderada.

Default `k=5` (coincide con tests).
"""
from __future__ import annotations

from typing import List, Tuple

from src.core.ports import RetrieverPort

__all__ = ["HybridRetriever"]


class HybridRetriever(RetrieverPort):
    """Combina un retriever denso y uno sparse mediante interpolación lineal.

    score_final = (1‑alpha)·score_dense + alpha·score_sparse
    """

    def __init__(self, *, dense: RetrieverPort, sparse: RetrieverPort, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0,1]")
        self.dense = dense
        self.sparse = sparse
        self.alpha = alpha

    # --------------------------------------------------------------
    def retrieve(self, query: str, k: int = 5) -> Tuple[List[int], List[float]]:
        d_ids, d_scores = self.dense.retrieve(query, k)
        s_ids, s_scores = self.sparse.retrieve(query, k)

        # Re‑ponderar
        d_scores = [s * (1 - self.alpha) for s in d_scores]
        s_scores = [s * self.alpha for s in s_scores]

        merged: dict[int, float] = {}
        for doc_id, score in zip(d_ids, d_scores):
            merged[doc_id] = merged.get(doc_id, 0.0) + score
        for doc_id, score in zip(s_ids, s_scores):
            merged[doc_id] = merged.get(doc_id, 0.0) + score

        # Orden descendente por score
        sorted_pairs = sorted(merged.items(), key=lambda t: t[1], reverse=True)[:k]
        ids, scores = zip(*sorted_pairs) if sorted_pairs else ([], [])
        return list(ids), list(scores)

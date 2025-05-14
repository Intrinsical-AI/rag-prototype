"""
File: src/adapters/retrieval/hybrid.py
Hybrid = combine dense + sparse scores (weighted sum).
"""

from typing import List, Tuple
from src.core.ports import RetrieverPort

class HybridRetriever(RetrieverPort):
    def __init__(
        self,
        dense: RetrieverPort,
        sparse: RetrieverPort,
        alpha: float = 0.5,   # 0=dense-only, 1=sparse-only
    ):
        self.dense = dense
        self.sparse = sparse
        self.alpha = alpha

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, k: int = 5) -> Tuple[List[int], List[float]]:
        ids_dense, sc_dense = self.dense.retrieve(query, k)
        ids_sparse, sc_sparse = self.sparse.retrieve(query, k)

        score_map: dict[int, float] = {}
        # sum ponderado
        for _id, s in zip(ids_dense, sc_dense):
            score_map[_id] = score_map.get(_id, 0) + (1 - self.alpha) * s
        for _id, s in zip(ids_sparse, sc_sparse):
            score_map[_id] = score_map.get(_id, 0) + self.alpha * s

        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:k]
        ids   = [doc_id for doc_id, _ in ranked]
        scores = [score for _, score in ranked]
        return ids, scores
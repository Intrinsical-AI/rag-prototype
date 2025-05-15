from __future__ import annotations

"""Hybrid Retriever: Combines Dense and Sparse (BM25) Retrievers.

Uses linear interpolation to combine results:

- final_score = (1 - alpha) * dense_score + alpha * sparse_score
```

```
"""

from typing import List, Tuple

from src.core.ports import RetrieverPort

__all__ = ["HybridRetriever"]


class HybridRetriever(RetrieverPort):
    """Hybrid retrieval combining dense and sparse retrievers.

    Parameters
    ----------
    dense : RetrieverPort
        Retriever using dense vector embeddings.
    sparse : RetrieverPort
        Retriever using sparse (BM25) scores.
    alpha : float, default 0.5
        Interpolation parameter (0 ≤ alpha ≤ 1). Higher alpha emphasizes sparse retrieval.
    """

    def __init__(
        self, *, dense: RetrieverPort, sparse: RetrieverPort, alpha: float = 0.5
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Parameter 'alpha' must be in [0, 1].")
        self.dense = dense
        self.sparse = sparse
        self.alpha = alpha

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[int], List[float]]:
        """Retrieve the top-k documents by combining dense and sparse scores.

        Parameters
        ----------
        query : str
            Query string.
        k : int, default 5
            Number of top documents to retrieve.

        Returns
        -------
        Tuple[List[int], List[float]]
            Parallel lists containing document IDs and combined scores.
        """
        dense_ids, dense_scores = self.dense.retrieve(query, k)
        sparse_ids, sparse_scores = self.sparse.retrieve(query, k)

        # Weight scores
        dense_scores = [score * (1 - self.alpha) for score in dense_scores]
        sparse_scores = [score * self.alpha for score in sparse_scores]

        # Merge scores
        combined_scores: dict[int, float] = {}

        for doc_id, score in zip(dense_ids, dense_scores):
            combined_scores[doc_id] = combined_scores.get(doc_id, 0.0) + score

        for doc_id, score in zip(sparse_ids, sparse_scores):
            combined_scores[doc_id] = combined_scores.get(doc_id, 0.0) + score

        # Sort by combined score (descending)
        sorted_results = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )[:k]

        ids, scores = zip(*sorted_results) if sorted_results else ([], [])

        return list(ids), list(scores)

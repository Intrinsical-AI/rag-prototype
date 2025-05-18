# src/infrastructure/retrieval/sparse_bm25.py

from typing import Sequence, Tuple

from src.core.domain.entities import Document
from src.core.ports import RetrieverPort


class HybridRetriever(RetrieverPort):
    def __init__(self, dense: RetrieverPort, sparse: RetrieverPort, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha debe estar en [0,1]")
        self.dense = dense
        self.sparse = sparse
        self.alpha = alpha

    def retrieve(
        self, query: str, k: int = 5
    ) -> Tuple[Sequence[Document], Sequence[float]]:
        dense_docs, dense_scores = self.dense.retrieve(query, k)
        sparse_docs, sparse_scores = self.sparse.retrieve(query, k)

        # Creamos mappings id->doc, id->score
        dense_map = {
            doc.id: (doc, score) for doc, score in zip(dense_docs, dense_scores)
        }
        sparse_map = {
            doc.id: (doc, score) for doc, score in zip(sparse_docs, sparse_scores)
        }
        # Unimos IDs
        all_ids = set(dense_map) | set(sparse_map)
        combined = []
        for doc_id in all_ids:
            d_score = dense_map.get(doc_id, (None, 0.0))[1]
            s_score = sparse_map.get(doc_id, (None, 0.0))[1]
            doc = (
                dense_map.get(doc_id, (None,))[0] or sparse_map.get(doc_id, (None,))[0]
            )
            if doc:
                score = (1 - self.alpha) * d_score + self.alpha * s_score
                combined.append((doc, score))
        # Ordenar por score descendente
        combined = sorted(combined, key=lambda t: t[1], reverse=True)[:k]
        docs, scores = zip(*combined) if combined else ([], [])
        return list(docs), list(scores)

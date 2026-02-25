# src/infrastructure/retrieval/sparse_bm25.py
"""
Sparse retriever using BM25.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from local_rag_backend.core.ports import DocumentRepoPort, RetrieverPort

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from local_rag_backend.core.domain.entities import Document
    from local_rag_backend.core.domain.types import DocId


class SparseBM25Retriever(RetrieverPort):
    """Sparse retriever using the BM25 algorithm."""

    def __init__(
        self,
        documents: Sequence[str],
        doc_ids: Sequence[DocId],
        doc_repo: DocumentRepoPort,
        *,
        preloaded_docs: Sequence[Document] | None = None,
    ):
        self.doc_ids = list(doc_ids)
        self.doc_repo = doc_repo
        self.bm25 = None
        self._tokenized_corpus = [self._tokenize(doc) for doc in documents] if documents else []
        if preloaded_docs is None:
            self._docs_by_id = {doc.id: doc for doc in doc_repo.get(self.doc_ids)}
        else:
            expected = set(self.doc_ids)
            self._docs_by_id = {doc.id: doc for doc in preloaded_docs if doc.id in expected}
            missing = [doc_id for doc_id in self.doc_ids if doc_id not in self._docs_by_id]
            if missing:
                self._docs_by_id.update({doc.id: doc for doc in doc_repo.get(missing)})

        if self._tokenized_corpus and any(self._tokenized_corpus):
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self._tokenized_corpus)

    def _ensure_docs_cache(self) -> dict[DocId, Document]:
        docs_by_id = getattr(self, "_docs_by_id", None)
        if docs_by_id is None:
            docs_by_id = {doc.id: doc for doc in self.doc_repo.get(self.doc_ids)}
            self._docs_by_id = docs_by_id
        return docs_by_id

    def _best_index_for_tied_top_score(
        self,
        ranked_indices: NDArray[np.intp],
        rank_scores: NDArray[np.float32],
        query_tokens: list[str],
    ) -> int:
        best = int(ranked_indices[0])
        best_score = float(rank_scores[best])
        tied = [int(i) for i in ranked_indices if float(rank_scores[int(i)]) == best_score]
        if len(tied) <= 1:
            return best

        tokenized_corpus = getattr(self, "_tokenized_corpus", [])
        if not tokenized_corpus:
            return min(tied)

        query_set = set(query_tokens)
        best_overlap = -1
        best_idx = min(tied)
        for idx in tied:
            tokens = tokenized_corpus[idx] if idx < len(tokenized_corpus) else []
            overlap = sum(1 for tok in tokens if tok in query_set)
            if overlap > best_overlap or (overlap == best_overlap and idx < best_idx):
                best_overlap = overlap
                best_idx = idx
        return best_idx

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Preprocess and tokenize text for BM25."""
        import re

        from local_rag_backend.core.services.text_processing import preprocess_text

        return re.findall(r"\w+", preprocess_text(text))

    def retrieve(self, query: str, k: int = 5) -> tuple[Sequence[Document], Sequence[float]]:
        """Retrieve documents using BM25 scores."""
        if k <= 0:
            return [], []
        if not self.bm25 or not query:
            return [], []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [], []

        doc_scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float32)
        if doc_scores.size == 0:
            return [], []
        k_eff = min(int(k), int(doc_scores.size))
        if k_eff <= 0:
            return [], []
        rank_scores = np.nan_to_num(doc_scores, nan=float("-inf"))
        ranked_indices = np.argsort(-rank_scores, kind="stable")
        if k_eff == 1:
            top_indices = np.array(
                [self._best_index_for_tied_top_score(ranked_indices, rank_scores, query_tokens)],
                dtype=np.int64,
            )
        else:
            top_indices = ranked_indices[:k_eff]

        retrieved_ids = [self.doc_ids[int(i)] for i in top_indices]
        scores = [float(rank_scores[int(i)]) for i in top_indices]

        if not scores:
            return [], []
        min_score, max_score = min(scores), max(scores)
        if max_score == min_score:
            normalized_scores = [1.0] * len(scores)
        else:
            normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]

        docs_by_id = self._ensure_docs_cache()
        ordered_docs = [docs_by_id[doc_id] for doc_id in retrieved_ids if doc_id in docs_by_id]
        score_by_id = {
            doc_id: score
            for doc_id, score in zip(retrieved_ids, normalized_scores, strict=False)
            if doc_id in docs_by_id
        }
        ordered_scores = [score_by_id[d.id] for d in ordered_docs]

        return ordered_docs, ordered_scores

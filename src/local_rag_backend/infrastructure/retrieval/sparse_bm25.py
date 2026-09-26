# src/infrastructure/retrieval/sparse_bm25.py
"""
Sparse retriever using BM25.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from local_rag_backend.core.domain.retrieval import (
    RetrievalRequest,
    RetrievalResult,
    retrieval_result_from_pairs,
)
from local_rag_backend.core.ports import DocumentRepoPort, RetrieverPort
from local_rag_backend.infrastructure.retrieval.scoring import normalize_min_max_scores

if TYPE_CHECKING:
    from collections.abc import Sequence

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

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Preprocess and tokenize text for BM25."""
        import re

        from local_rag_backend.core.services.text_processing import preprocess_text

        return re.findall(r"\w+", preprocess_text(text))

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Retrieve documents using BM25 scores."""
        if self.bm25 is None:
            return RetrievalResult(items=(), mode_used="sparse", backend_used="local_bm25")

        query_tokens = self._tokenize(request.query)
        if not query_tokens:
            return RetrievalResult(items=(), mode_used="sparse", backend_used="local_bm25")

        doc_scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float32)
        if doc_scores.size == 0:
            return RetrievalResult(items=(), mode_used="sparse", backend_used="local_bm25")
        k_eff = min(int(request.top_k), int(doc_scores.size))
        rank_scores = np.nan_to_num(doc_scores, nan=float("-inf"))
        query_set = set(query_tokens)
        overlaps = [
            sum(token in query_set for token in tokens) for tokens in self._tokenized_corpus
        ]
        top_indices = sorted(
            range(len(rank_scores)),
            key=lambda index: (-float(rank_scores[index]), -overlaps[index], index),
        )[:k_eff]

        retrieved_ids = [self.doc_ids[int(i)] for i in top_indices]
        scores = [float(rank_scores[int(i)]) for i in top_indices]

        normalized_scores = normalize_min_max_scores(scores, flat_value=0.0, singleton_value=1.0)

        docs_by_id = self._docs_by_id
        ordered_docs = [docs_by_id[doc_id] for doc_id in retrieved_ids if doc_id in docs_by_id]
        score_by_id = {
            doc_id: score
            for doc_id, score in zip(retrieved_ids, normalized_scores, strict=False)
            if doc_id in docs_by_id
        }
        ordered_scores = [score_by_id[d.id] for d in ordered_docs]

        return retrieval_result_from_pairs(
            docs=ordered_docs,
            scores=ordered_scores,
            mode_used="sparse",
            backend_used="local_bm25",
            stage="sparse",
        )

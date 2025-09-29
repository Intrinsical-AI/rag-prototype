# src/infrastructure/retrieval/sparse_bm25.py
"""
Sparse retriever using BM25.
"""

from __future__ import annotations

from collections.abc import Sequence

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.ports import DocumentRepoPort, RetrieverPort


class SparseBM25Retriever(RetrieverPort):
    """Sparse retriever using the BM25 algorithm."""

    def __init__(
        self, documents: Sequence[str], doc_ids: Sequence[int], doc_repo: DocumentRepoPort
    ):
        self.doc_ids = doc_ids
        self.doc_repo = doc_repo
        self.bm25 = None

        if documents:
            tokenized_corpus = [self._tokenize(doc) for doc in documents]
            if any(tokenized_corpus):
                from rank_bm25 import BM25Okapi

                self.bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Preprocess and tokenize text for BM25."""
        import re

        from local_rag_backend.utils import preprocess_text

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

        doc_scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:k]

        retrieved_ids = [self.doc_ids[i] for i in top_indices]
        scores = [doc_scores[i] for i in top_indices]

        # Normalize scores to [0, 1]
        if not scores:
            return [], []
        min_score, max_score = min(scores), max(scores)
        if max_score == min_score:
            normalized_scores = [1.0] * len(scores)
        else:
            normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]

        docs = self.doc_repo.get(retrieved_ids)
        docs_by_id = {doc.id: doc for doc in docs}

        # Ensure correct order
        ordered_docs = [docs_by_id[doc_id] for doc_id in retrieved_ids if doc_id in docs_by_id]

        return ordered_docs, normalized_scores

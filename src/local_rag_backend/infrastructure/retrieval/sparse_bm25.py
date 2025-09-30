"""
RAG Prototype - Intrinsical-AI (c) 2025
Author: Pablo Pintor
License: MIT

Module: Sparse BM25 Retriever
Purpose: Implements sparse keyword-based document retrieval using the BM25 algorithm.
         Provides traditional information retrieval capabilities based on term frequency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_backend.core.ports import DocumentRepoPort, RetrieverPort

if TYPE_CHECKING:
    from collections.abc import Sequence

    from local_rag_backend.core.domain.entities import Document


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
        """Retrieve documents using BM25 scores.
        
        Args:
            query: The search query text to find similar documents for
            k: Maximum number of documents to retrieve
            
        Returns:
            Tuple containing:
                - List of Document objects ordered by relevance
                - List of similarity scores corresponding to each document
                
        Note:
            Returns empty lists if k <= 0, query is invalid, or no BM25 index exists.
        """
        # --- Input Validation ---
        if k <= 0:
            return [], []
        if not self.bm25:
            return [], []
        if not self._is_valid_query(query):
            return [], []

        # --- Query Tokenization ---
        normalized_query = query.strip()
        query_tokens = self._tokenize(normalized_query)
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

        # Ensure correct order and maintain docs-scores consistency
        ordered_docs = []
        filtered_scores = []
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in docs_by_id:
                ordered_docs.append(docs_by_id[doc_id])
                filtered_scores.append(normalized_scores[i])

        return ordered_docs, filtered_scores

    def _is_valid_query(self, query: str | None) -> bool:
        """Validate that query is suitable for processing.
        
        Args:
            query: Query string to validate
            
        Returns:
            True if query is valid, False otherwise
        """
        if query is None:
            return False
        if not isinstance(query, str):
            return False
        if not query.strip():
            return False
        return True

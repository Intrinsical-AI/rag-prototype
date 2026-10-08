# src/infrastructure/retrieval/sparse_bm25.py
"""
Sparse retriever using BM25.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

    from numpy.typing import NDArray

    from local_rag_backend.core.domain.entities import Document
    from local_rag_backend.core.domain.types import DocId

    Postings = dict[str, tuple[NDArray[np.int64], NDArray[np.int64]]]


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

        # Per-term postings over the same BM25Okapi state, filled as terms are first queried.
        self._indexed_bm25: Any = None
        self._postings: Postings = {}
        self._length_norm: NDArray[np.float64] = np.zeros(0)
        if self._tokenized_corpus and any(self._tokenized_corpus):
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self._tokenized_corpus)
            self._indexed_bm25 = bm25 = self.bm25
            doc_len = np.array(bm25.doc_len)
            # Same expression and evaluation order as BM25Okapi.get_scores.
            self._length_norm = bm25.k1 * (1 - bm25.b + bm25.b * doc_len / bm25.avgdl)

    def _posting(self, term: str) -> tuple[NDArray[np.int64], NDArray[np.int64]] | None:
        """Ascending indices of documents containing term, with its frequencies (cached)."""
        posting = self._postings.get(term)
        if posting is None and term in self._indexed_bm25.idf:
            pairs = [
                (index, freq)
                for index, frequencies in enumerate(self._indexed_bm25.doc_freqs)
                if (freq := frequencies.get(term))
            ]
            columns = np.array(pairs, dtype=np.int64)
            posting = (columns[:, 0].copy(), columns[:, 1].copy())
            self._postings[term] = posting
        return posting

    def _bm25_scores(self, query_tokens: list[str]) -> NDArray[np.float64]:
        """BM25Okapi.get_scores restricted to each term's postings; bit-identical result."""
        bm25 = self._indexed_bm25
        scores = np.zeros(bm25.corpus_size)
        numerator_scale = bm25.k1 + 1
        for term in query_tokens:
            idf = bm25.idf.get(term) or 0
            posting = self._posting(term) if idf else None
            if posting is None:
                continue  # get_scores adds only (signed) zeros for this term
            doc_idx, freqs = posting
            scores[doc_idx] += idf * (
                freqs * numerator_scale / (freqs + self._length_norm[doc_idx])
            )
        return scores

    def _query_overlaps(self, query_tokens: list[str]) -> NDArray[np.int64]:
        """Per document, how many of its tokens belong to the query's token set."""
        overlaps = np.zeros(self._indexed_bm25.corpus_size, dtype=np.int64)
        for term in set(query_tokens):
            posting = self._posting(term)
            if posting is not None:
                overlaps[posting[0]] += posting[1]
        return overlaps

    def _rank(self, query_tokens: list[str], top_k: int) -> tuple[list[int], list[float]] | None:
        """Top-k document indices by (-bm25, -overlap, index), with their float32 scores."""
        bm25 = self.bm25
        if bm25 is None:
            return None
        if bm25 is self._indexed_bm25:
            doc_scores = self._bm25_scores(query_tokens).astype(np.float32)
            overlaps = self._query_overlaps(query_tokens)
        else:  # bm25 was replaced after construction: score through its public API.
            doc_scores = np.asarray(bm25.get_scores(query_tokens), dtype=np.float32)
            query_set = set(query_tokens)
            overlaps = np.asarray(
                [sum(token in query_set for token in tokens) for tokens in self._tokenized_corpus],
                dtype=np.int64,
            )
        if doc_scores.size == 0:
            return None
        k_eff = min(int(top_k), int(doc_scores.size))
        rank_scores = np.nan_to_num(doc_scores, nan=float("-inf"))
        # lexsort is stable: equal (score, overlap) keys keep ascending index order.
        top_indices = np.lexsort((-overlaps, -rank_scores))[:k_eff]
        return [int(i) for i in top_indices], [float(rank_scores[i]) for i in top_indices]

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

        ranked = self._rank(query_tokens, request.top_k)
        if ranked is None:
            return RetrievalResult(items=(), mode_used="sparse", backend_used="local_bm25")
        top_indices, scores = ranked
        retrieved_ids = [self.doc_ids[i] for i in top_indices]

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

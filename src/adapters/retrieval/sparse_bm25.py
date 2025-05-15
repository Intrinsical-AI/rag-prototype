# src/adapters/retrieval/sparse_bm25.py
from typing import List, Tuple
import re

from src.core.ports import RetrieverPort
from rank_bm25 import BM25Okapi
from src.utils import preprocess_text

import logging

logger = logging.getLogger(__name__)


class SparseBM25Retriever(RetrieverPort):
    def __init__(self, documents: List[str], doc_ids: List[int]):
        self.doc_ids = doc_ids
        self.bm25 = None
        self.corpus_is_empty = not documents  # flag

        if not self.corpus_is_empty:
            tokenized_corpus = [self._tok(d) for d in documents]
            # Only init BM25 if tokenized corpus have content - if not, rank_bm25 might fail
            if any(tokenized_corpus):
                try:
                    self.bm25 = BM25Okapi(tokenized_corpus)
                except ZeroDivisionError:
                    # safeguard - testing feedback
                    logger.warning(
                        "BM25Okapi ZeroDivisionError despite non-empty tokenized corpus. BM25 will not be initialized.",
                        exc_info=True,
                    )  # CAMBIO A INGLÉS

                    self.corpus_is_empty = True
                    self.bm25 = None
            else:
                logger.warning(
                    "WARNING: Tokenized corpus it's empty. BM25 will not be initialized."
                )
                self.corpus_is_empty = True
        else:
            logger.warning(
                "WARNING: Document corpus it's empty. BM25 will not be initialized."
            )

    @staticmethod
    def _tok(text: str) -> List[str]:
        return re.findall(r"\w+", preprocess_text(text))

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[int], List[float]]:
        if self.corpus_is_empty or self.bm25 is None:
            return [], []

        query_tokens = self._tok(query)
        if not query_tokens:
            return [], []

        doc_scores = self.bm25.get_scores(query_tokens)

        # k top scores or all docs
        num_docs_to_return = min(k, len(doc_scores))

        top_indices = sorted(
            range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True
        )[:num_docs_to_return]

        retrieved_ids = [self.doc_ids[i] for i in top_indices]
        retrieved_scores_raw = [doc_scores[i] for i in top_indices]

        # Norm between [0,1]
        if not retrieved_scores_raw:
            return retrieved_ids, []

        min_score = min(retrieved_scores_raw)
        max_score = max(retrieved_scores_raw)

        if max_score == min_score:
            # if only 1 score retrieved, or all equals
            normalized_scores = [0.0 if max_score == 0 else 1.0] * len(
                retrieved_scores_raw
            )
        else:
            normalized_scores = [
                (s - min_score) / (max_score - min_score) for s in retrieved_scores_raw
            ]

        return retrieved_ids, normalized_scores

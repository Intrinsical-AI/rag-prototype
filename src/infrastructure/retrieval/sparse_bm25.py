# src/infrastructure/retrieval/sparse_bm25.py

from typing import Sequence, Tuple

from src.core.domain.entities import Document
from src.core.ports import RetrieverPort


class SparseBM25Retriever(RetrieverPort):
    def __init__(self, documents, doc_ids, doc_repo):
        self.doc_ids = doc_ids
        self.doc_repo = doc_repo
        self.bm25 = None
        self.corpus_is_empty = not documents
        if not self.corpus_is_empty:
            tokenized_corpus = [self._tok(d) for d in documents]
            if any(tokenized_corpus):
                from rank_bm25 import BM25Okapi

                self.bm25 = BM25Okapi(tokenized_corpus)
            else:
                self.corpus_is_empty = True
        else:
            self.corpus_is_empty = True

    @staticmethod
    def _tok(text: str):
        import re

        from src.utils import preprocess_text

        return re.findall(r"\w+", preprocess_text(text))

    def retrieve(
        self, query: str, k: int = 5
    ) -> Tuple[Sequence[Document], Sequence[float]]:
        if self.corpus_is_empty or self.bm25 is None:
            return [], []
        query_tokens = self._tok(query)
        if not query_tokens:
            return [], []
        doc_scores = self.bm25.get_scores(query_tokens)
        num_docs_to_return = min(k, len(doc_scores))
        top_indices = sorted(
            range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True
        )[:num_docs_to_return]
        retrieved_ids = [self.doc_ids[i] for i in top_indices]
        retrieved_scores_raw = [doc_scores[i] for i in top_indices]
        # Normalizamos
        if not retrieved_scores_raw:
            return [], []
        min_score = min(retrieved_scores_raw)
        max_score = max(retrieved_scores_raw)
        if max_score == min_score:
            normalized_scores = [0.0 if max_score == 0 else 1.0] * len(
                retrieved_scores_raw
            )
        else:
            normalized_scores = [
                (s - min_score) / (max_score - min_score) for s in retrieved_scores_raw
            ]
        docs = self.doc_repo.get(retrieved_ids)
        # Map id->score para match exacto con los docs (si hay desfase)
        id_to_score = dict(zip(retrieved_ids, normalized_scores))
        final_docs, final_scores = [], []
        for doc in docs:
            if doc.id in id_to_score:
                final_docs.append(doc)
                final_scores.append(float(id_to_score[doc.id]))
        return final_docs, final_scores

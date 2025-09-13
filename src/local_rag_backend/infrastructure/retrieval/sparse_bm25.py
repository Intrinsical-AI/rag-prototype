# src/infrastructure/retrieval/sparse_bm25.py

from collections.abc import Sequence

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.ports import DocumentRepoPort, RetrieverPort
from local_rag_backend.utils import normalize_similarities_from_distances


class SparseBM25Retriever(RetrieverPort):
    def __init__(
        self, documents: Sequence[str], doc_ids: Sequence[int], doc_repo: DocumentRepoPort
    ):
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
    def _tok(text: str) -> list[str]:
        import re

        from local_rag_backend.utils import preprocess_text

        return re.findall(r"\w+", preprocess_text(text))

    def retrieve(self, query: str, k: int = 5) -> tuple[Sequence[Document], Sequence[float]]:
        if self.corpus_is_empty or self.bm25 is None:
            return [], []
        query_tokens = self._tok(query)
        if not query_tokens:
            return [], []
        doc_scores = self.bm25.get_scores(query_tokens)
        num_docs_to_return = min(k, len(doc_scores))
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[
            :num_docs_to_return
        ]
        retrieved_ids = [self.doc_ids[i] for i in top_indices]
        retrieved_scores_raw = [doc_scores[i] for i in top_indices]
        # Normalize using the helper function (treating BM25 scores as "distances" - higher is better, so invert)
        if not retrieved_scores_raw:
            return [], []
        # BM25 scores are similarities (higher is better), but our helper expects distances (lower is better)
        # So we invert them by subtracting from max to convert to distance-like values
        max_raw = max(retrieved_scores_raw)
        distance_like = [max_raw - s for s in retrieved_scores_raw]
        normalized_scores = normalize_similarities_from_distances(distance_like)
        # Fix ordering bug: reconstruct (doc, score) in the order of retrieved indices
        docs_by_id = {d.id: d for d in self.doc_repo.get(retrieved_ids)}
        ordered = []
        for doc_id, sc in zip(retrieved_ids, normalized_scores, strict=False):
            doc = docs_by_id.get(doc_id)
            if doc is not None:
                ordered.append((doc, float(sc)))
        docs, scores = zip(*ordered, strict=False) if ordered else ([], [])
        return list(docs), list(scores)

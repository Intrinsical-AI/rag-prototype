# src/infrastructure/retrieval/dense_faiss.py

from collections.abc import Sequence
from typing import TYPE_CHECKING

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.ports import DocumentRepoPort, EmbedderPort, RetrieverPort

if TYPE_CHECKING:
    from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage


class DenseFaissRetriever(RetrieverPort):
    def __init__(
        self, embedder: EmbedderPort, faiss_index: "FaissVectorStorage", doc_repo: DocumentRepoPort
    ):
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.doc_repo = doc_repo

    def retrieve(self, query: str, k: int = 5) -> tuple[Sequence[Document], Sequence[float]]:
        if k <= 0:
            return [], []
        q_vec = self.embedder.embed([query])[0]
        # FAISS IndexFlatL2 returns squared L2 distances (lower is better)
        idxs, dists = self.faiss_index.search(q_vec, k)
        # Convert distances -> similarities in [0,1] (monotonic): sim = 1/(1+d)
        # Filter out invalid entries (-1) before normalization
        valid_pairs = [(i, d) for i, d in zip(idxs, dists, strict=False) if i != -1]
        if valid_pairs:
            valid_dists = [d for _, d in valid_pairs]
            sims_raw = [1.0 / (1.0 + float(d)) for d in valid_dists]
            # Normalize locally to [0,1] similar to BM25 retriever
            min_s, max_s = min(sims_raw), max(sims_raw)
            if max_s == min_s:
                sims = [0.0 if max_s == 0 else 1.0] * len(sims_raw)
            else:
                sims = [(s - min_s) / (max_s - min_s) for s in sims_raw]
        else:
            sims = []
        # Fix ordering bug: reconstruct (doc, score) in the order of retrieved indices
        # Fix duplicate SQL fetch: get docs once and reuse
        ids = [self.faiss_index.id_map[i] for i, _ in valid_pairs]
        docs = self.doc_repo.get(ids)
        docs_by_id = {d.id: d for d in docs}
        ordered = []
        for (i, _), sim in zip(valid_pairs, sims, strict=False):
            real_id = self.faiss_index.id_map[i]
            doc = docs_by_id.get(real_id)
            if doc is not None:
                ordered.append((doc, float(sim)))
        docs, scores = zip(*ordered, strict=False) if ordered else ([], [])
        return list(docs), list(scores)

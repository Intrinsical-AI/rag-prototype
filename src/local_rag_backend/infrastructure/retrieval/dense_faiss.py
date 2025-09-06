# src/infrastructure/retrieval/dense_faiss.py

from typing import Sequence, Tuple

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.ports import RetrieverPort


class DenseFaissRetriever(RetrieverPort):
    def __init__(self, embedder, faiss_index, doc_repo):
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.doc_repo = doc_repo

    def retrieve(
        self, query: str, k: int = 5
    ) -> Tuple[Sequence[Document], Sequence[float]]:
        if k <= 0:
            return [], []
        q_vec = self.embedder.embed([query])[0]
        # FAISS IndexFlatL2 returns squared L2 distances (lower is better)
        idxs, dists = self.faiss_index.search(q_vec, k)
        ids = [self.faiss_index.id_map[i] for i in idxs if i != -1]
        docs = self.doc_repo.get(ids)
        # Convert distances -> similarities in [0,1] (monotonic): sim = 1/(1+d)
        sims_raw = [1.0 / (1.0 + float(d)) for d in dists]
        # Normalize locally to [0,1] similar to BM25 retriever
        if sims_raw:
            min_s, max_s = min(sims_raw), max(sims_raw)
            if max_s == min_s:
                sims = [0.0 if max_s == 0 else 1.0] * len(sims_raw)
            else:
                sims = [(s - min_s) / (max_s - min_s) for s in sims_raw]
        else:
            sims = []
        # Map id -> normalized score, guarding invalid indices (avoid misalignment)
        id_to_score = {}
        for i, sim in zip(idxs, sims):
            if i != -1:
                real_id = self.faiss_index.id_map[i]
                id_to_score[real_id] = float(sim)
        final_docs, final_scores = [], []
        for doc in docs:
            if doc.id in id_to_score:
                final_docs.append(doc)
                final_scores.append(float(id_to_score[doc.id]))
        return final_docs, final_scores

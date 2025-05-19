# src/infrastructure/retrieval/dense_faiss.py

from typing import Sequence, Tuple

from src.core.domain.entities import Document
from src.core.ports import RetrieverPort


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
        idxs, scores = self.faiss_index.search(q_vec, k)
        ids = [self.faiss_index.id_map[i] for i in idxs if i != -1]
        docs = self.doc_repo.get(ids)
        # Cuidado: puede haber desfase si algún id no existe, así que cruzamos id con doc.
        id_to_score = {
            id_: score for id_, i, score in zip(ids, idxs, scores) if i != -1
        }
        final_docs, final_scores = [], []
        for doc in docs:
            if doc.id in id_to_score:
                final_docs.append(doc)
                final_scores.append(float(id_to_score[doc.id]))
        return final_docs, final_scores

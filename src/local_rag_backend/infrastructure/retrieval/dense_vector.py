"""Dense retriever using a vector repository."""

from __future__ import annotations

from local_rag_backend.core.domain.retrieval import (
    RetrievalRequest,
    RetrievalResult,
    retrieval_result_from_pairs,
)
from local_rag_backend.core.ports import (
    DocumentRepoPort,
    EmbedderPort,
    RetrieverPort,
    VectorRepoPort,
)


class DenseVectorRetriever(RetrieverPort):
    """Dense retriever using vector similarity search."""

    def __init__(
        self, embedder: EmbedderPort, vector_repo: VectorRepoPort, doc_repo: DocumentRepoPort
    ):
        self.embedder = embedder
        self.vector_repo = vector_repo
        self.doc_repo = doc_repo

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        if request.top_k <= 0:
            return RetrievalResult(items=(), mode_used="dense", backend_used="legacy_dense")

        query_embedding = self.embedder.embed([request.query])[0]
        id_score_pairs = self.vector_repo.similar(query_embedding, request.top_k)
        if not id_score_pairs:
            return RetrievalResult(items=(), mode_used="dense", backend_used="legacy_dense")

        doc_ids, scores = zip(*id_score_pairs, strict=False)
        docs = self.doc_repo.get(list(doc_ids))

        docs_by_id = {doc.id: doc for doc in docs}
        ordered_pairs = list(zip(doc_ids, scores, strict=False))
        ordered_docs = [docs_by_id[doc_id] for doc_id, _ in ordered_pairs if doc_id in docs_by_id]
        ordered_scores = [score for doc_id, score in ordered_pairs if doc_id in docs_by_id]

        return retrieval_result_from_pairs(
            docs=ordered_docs,
            scores=ordered_scores,
            mode_used="dense",
            backend_used="legacy_dense",
            stage="dense",
        )

"""Dense retriever using a vector repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_backend.core.domain.retrieval import (
    RetrievalRequest,
    RetrievalResult,
    RetrievedDoc,
    document_matches_filters,
)
from local_rag_backend.core.ports import (
    DocumentRepoPort,
    EmbedderPort,
    RetrieverPort,
    VectorRepoPort,
)

if TYPE_CHECKING:
    from local_rag_backend.core.domain.entities import Document
    from local_rag_backend.core.domain.types import DocId


class DenseVectorRetriever(RetrieverPort):
    """Dense retriever using vector similarity search."""

    def __init__(
        self, embedder: EmbedderPort, vector_repo: VectorRepoPort, doc_repo: DocumentRepoPort
    ):
        self.embedder = embedder
        self.vector_repo = vector_repo
        self.doc_repo = doc_repo

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        query_embedding = self.embedder.embed([request.query])[0]
        window = max(request.top_k, request.candidate_k or request.top_k)
        docs_by_id: dict[DocId, Document] = {}
        examined_ids: set[DocId] = set()
        while True:
            pairs = self.vector_repo.similar(query_embedding, window)
            new_ids = [doc_id for doc_id, _score in pairs if doc_id not in examined_ids]
            if new_ids:
                docs_by_id.update((doc.id, doc) for doc in self.doc_repo.get(new_ids))
                examined_ids.update(new_ids)
            eligible = [
                RetrievedDoc(document=doc, score=float(score), stage="dense")
                for doc_id, score in pairs
                if (doc := docs_by_id.get(doc_id)) is not None
                and document_matches_filters(doc, request.filters)
            ]
            if len(eligible) >= request.top_k or len(pairs) < window:
                break
            total = self.vector_repo.ntotal
            if window >= total:
                break
            window = min(window * 2, total)

        # Scores belong to the final candidate window. Thresholds must not trigger
        # further overfetch: min-max normalization itself depends on that window.
        if request.min_score is not None:
            eligible = [item for item in eligible if item.score >= request.min_score]
        return RetrievalResult(
            items=tuple(eligible[: request.top_k]),
            mode_used="dense",
            backend_used="local_vector",
            candidate_count=len(pairs),
        )

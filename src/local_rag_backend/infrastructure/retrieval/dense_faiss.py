# src/infrastructure/retrieval/dense_faiss.py
"""
Dense retriever using FAISS for vector search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_backend.core.ports import (
    DocumentRepoPort,
    EmbedderPort,
    RetrieverPort,
    VectorRepoPort,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from local_rag_backend.core.domain.entities import Document


class DenseFaissRetriever(RetrieverPort):
    """Dense retriever using FAISS for vector search."""

    def __init__(
        self, embedder: EmbedderPort, faiss_index: VectorRepoPort, doc_repo: DocumentRepoPort
    ):
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.doc_repo = doc_repo

    def retrieve(self, query: str, k: int = 5) -> tuple[Sequence[Document], Sequence[float]]:
        """Retrieve documents based on dense vector similarity."""
        if k <= 0:
            return [], []

        query_embedding = self.embedder.embed([query])[0]
        id_score_pairs = self.faiss_index.similar(query_embedding, k)
        if not id_score_pairs:
            return [], []

        doc_ids, scores = zip(*id_score_pairs, strict=False)
        docs = self.doc_repo.get(list(doc_ids))

        # Ensure correct order and pairing
        docs_by_id = {doc.id: doc for doc in docs}
        ordered_docs = [docs_by_id[doc_id] for doc_id in doc_ids if doc_id in docs_by_id]

        return ordered_docs, list(scores)

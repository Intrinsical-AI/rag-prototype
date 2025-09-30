"""
Intrinsical-AI RAG Prototype
Copyright (c) 2025 Intrinsical-AI

Module: Dense FAISS Retriever
Purpose: Implements dense vector-based document retrieval using FAISS for similarity search.
         Provides semantic search capabilities through embedding-based matching.
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
    """Dense vector-based document retriever using FAISS for similarity search.

    This retriever implements semantic search by:
    1. Converting queries to dense vector embeddings
    2. Performing similarity search against indexed document embeddings
    3. Retrieving and ranking documents based on vector similarity scores

    The retriever maintains the order of results as returned by FAISS to preserve
    relevance ranking based on cosine similarity or L2 distance.
    """

    def __init__(
        self, embedder: EmbedderPort, faiss_index: VectorRepoPort, doc_repo: DocumentRepoPort
    ):
        """Initialize the dense retriever with required dependencies.

        Args:
            embedder: Service for converting text to vector embeddings
            faiss_index: FAISS-based vector storage for similarity search
            doc_repo: Repository for retrieving document content by ID
        """
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.doc_repo = doc_repo

    def retrieve(self, query: str, k: int = 5) -> tuple[Sequence[Document], Sequence[float]]:
        """Retrieve documents based on dense vector similarity to the query.

        Args:
            query: The search query text to find similar documents for
            k: Maximum number of documents to retrieve

        Returns:
            Tuple containing:
                - List of Document objects ordered by relevance
                - List of similarity scores corresponding to each document

        Note:
            Returns empty lists if k <= 0 or no similar documents are found.
            Documents are returned in the same order as FAISS similarity results.
        """
        if k <= 0:
            return [], []

        # --- Query Embedding Phase ---
        query_embedding = self.embedder.embed([query])[0]

        # --- Vector Similarity Search ---
        id_score_pairs = self.faiss_index.similar(query_embedding, k)
        if not id_score_pairs:
            return [], []

        # --- Document Retrieval Phase ---
        doc_ids, scores = zip(*id_score_pairs, strict=False)
        docs = self.doc_repo.get(list(doc_ids))

        # --- Maintain Relevance Order ---
        # Ensure documents are returned in the same order as FAISS results
        docs_by_id = {doc.id: doc for doc in docs}
        ordered_docs = [docs_by_id[doc_id] for doc_id in doc_ids if doc_id in docs_by_id]

        return ordered_docs, list(scores)

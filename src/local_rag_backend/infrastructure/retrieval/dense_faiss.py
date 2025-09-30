"""
RAG Prototype - Intrinsical-AI (c) 2025
Author: Pablo Pintor
License: MIT

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

        Raises:
            ValueError: If query is invalid (None, empty, or whitespace-only)
            RuntimeError: If embedding generation fails

        Note:
            Returns empty lists if k <= 0 or no similar documents are found.
            Documents are returned in the same order as FAISS similarity results.
        """
        # --- Input Validation ---
        if k <= 0:
            return [], []
            
        if not self._is_valid_query(query):
            return [], []

        try:
            # --- Query Embedding Phase ---
            normalized_query = query.strip()
            embeddings = self.embedder.embed([normalized_query])
            
            if not embeddings:
                # Embedder returned empty list - treat as no results
                return [], []
                
            query_embedding = embeddings[0]

            # --- Vector Similarity Search ---
            id_score_pairs = self.faiss_index.similar(query_embedding, k)
            if not id_score_pairs:
                return [], []

            # --- Document Retrieval Phase ---
            doc_ids, scores = zip(*id_score_pairs, strict=False)
            docs = self.doc_repo.get(list(doc_ids))

            # --- Maintain Relevance Order and Docs-Scores Consistency ---
            docs_by_id = {doc.id: doc for doc in docs}
            ordered_docs = []
            filtered_scores = []
            for i, doc_id in enumerate(doc_ids):
                if doc_id in docs_by_id:
                    ordered_docs.append(docs_by_id[doc_id])
                    filtered_scores.append(scores[i])

            return ordered_docs, filtered_scores
            
        except Exception as e:
            # Log the error but return empty results rather than crashing
            # This ensures the retrieval system remains robust
            raise RuntimeError(f"Dense retrieval failed for query '{query}': {e}") from e

    def _is_valid_query(self, query: str | None) -> bool:
        """Validate that query is suitable for processing.
        
        Args:
            query: Query string to validate
            
        Returns:
            True if query is valid, False otherwise
        """
        if query is None:
            return False
        if not isinstance(query, str):
            return False
        if not query.strip():
            return False
        return True

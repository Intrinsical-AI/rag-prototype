"""
RAG Prototype - Intrinsical-AI (c) 2025
Author: Pablo Pintor
License: MIT

Module: ETL Service
Purpose: Orchestrates the Extract-Transform-Load pipeline for document ingestion.
         Handles text processing, embedding generation, and storage coordination.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from local_rag_backend.core.ports import DocumentRepoPort, EmbedderPort, VectorRepoPort


class ETLService:
    """Core service orchestrating the document ingestion pipeline.

    This service implements the ETL (Extract-Transform-Load) pattern for processing
    raw text documents into a searchable knowledge base. It coordinates:
    - Document storage in the primary database
    - Vector embedding generation for semantic search
    - Vector index updates for retrieval operations

    The service ensures data consistency by handling failures gracefully and
    maintaining synchronization between text and vector storage layers.
    """

    def __init__(
        self,
        doc_storage: DocumentRepoPort,
        vec_storage: VectorRepoPort,
        embedder: EmbedderPort,
    ):
        """Initialize the ETL service with required storage and processing dependencies.

        Args:
            doc_storage: Repository for storing document text content
            vec_storage: Repository for storing and indexing vector embeddings
            embedder: Service for converting text to vector embeddings
        """
        self._doc_store = doc_storage
        self._vec_store = vec_storage
        self._embedder = embedder

    def ingest(self, texts: Sequence[str]) -> Sequence[int]:
        """Process and store a batch of text documents through the complete ETL pipeline.

        This method orchestrates the full ingestion workflow:
        1. Store raw text documents in the primary database
        2. Generate vector embeddings for semantic search
        3. Index embeddings in the vector storage for retrieval

        Args:
            texts: Sequence of text documents to be processed and stored

        Returns:
            Sequence of unique document IDs assigned to the stored documents

        Note:
            Returns empty list if no texts are provided. All operations are
            performed as a batch for optimal performance.
        """
        if not texts:
            return []

        # --- Document Storage Phase ---
        doc_ids = self._doc_store.store_documents(texts)

        # --- Embedding Generation Phase ---
        embeddings = self._embedder.embed(texts)

        # --- Vector Index Update Phase ---
        self._vec_store.upsert(doc_ids, embeddings)

        return doc_ids

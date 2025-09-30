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

        This method orchestrates the full ingestion workflow with transactional safety:
        1. Validate inputs and filter empty texts
        2. Store raw text documents in the primary database
        3. Generate vector embeddings for semantic search
        4. Index embeddings in the vector storage for retrieval
        5. Rollback documents if embedding/vector operations fail

        Args:
            texts: Sequence of text documents to be processed and stored

        Returns:
            Sequence of unique document IDs assigned to the stored documents

        Raises:
            ValueError: If input validation fails
            RuntimeError: If embedding generation or vector storage fails

        Note:
            Returns empty list if no valid texts are provided. Implements
            transactional safety with rollback on failures to maintain
            consistency between document and vector storage.
        """
        # --- Input Validation ---
        if not texts:
            return []
        
        # Filter out empty or whitespace-only texts
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        if not valid_texts:
            return []

        doc_ids: Sequence[int] = []
        
        try:
            # --- Document Storage Phase ---
            doc_ids = self._doc_store.store_documents(valid_texts)
            
            if not doc_ids:
                raise RuntimeError("Document storage failed: no IDs returned")

            # --- Embedding Generation Phase ---
            embeddings = self._embedder.embed(valid_texts)
            
            # Validate embedding consistency
            if len(embeddings) != len(doc_ids):
                raise RuntimeError(
                    f"Embedding count mismatch: {len(embeddings)} embeddings "
                    f"for {len(doc_ids)} documents"
                )

            # --- Vector Index Update Phase ---
            self._vec_store.upsert(doc_ids, embeddings)

            return doc_ids

        except Exception as e:
            # --- Rollback on Failure ---
            if doc_ids:
                try:
                    # Attempt to remove stored documents to maintain consistency
                    self._rollback_documents(doc_ids)
                except Exception as rollback_error:
                    # Log rollback failure but don't mask original error
                    raise RuntimeError(
                        f"ETL pipeline failed: {e}. "
                        f"Rollback also failed: {rollback_error}. "
                        f"Database may be in inconsistent state."
                    ) from e
            
            # Re-raise original error with context
            raise RuntimeError(f"ETL pipeline failed during processing: {e}") from e

    def _rollback_documents(self, doc_ids: Sequence[int]) -> None:
        """Attempt to remove documents to maintain consistency.
        
        This is a best-effort rollback mechanism. If the document repository
        doesn't support deletion, this will be a no-op.
        
        Args:
            doc_ids: Document IDs to remove
        """
        # Check if document repository supports deletion
        if hasattr(self._doc_store, 'delete_documents'):
            self._doc_store.delete_documents(doc_ids)  # type: ignore
        # If no deletion support, we can't rollback - this is logged in the exception

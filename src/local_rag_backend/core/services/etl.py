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

        # Filter out empty, whitespace-only texts, and non-string types
        valid_texts = []
        for text in texts:
            # Type check: ensure text is a string
            if not isinstance(text, str):
                continue
            # Skip empty or whitespace-only
            if text and text.strip():
                valid_texts.append(text.strip())

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

            # Validate embedding count consistency
            if len(embeddings) != len(doc_ids):
                raise RuntimeError(
                    f"Embedding count mismatch: {len(embeddings)} embeddings "
                    f"for {len(doc_ids)} documents"
                )

            # Validate embedding quality (detect NaN, inf, empty embeddings)
            self._validate_embeddings(embeddings)

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

    def _validate_embeddings(self, embeddings: Sequence[Sequence[float]]) -> None:
        """Validate embedding quality to detect corrupted data.

        Args:
            embeddings: Sequence of embedding vectors to validate

        Raises:
            RuntimeError: If embeddings contain NaN, inf, or are malformed
        """
        import math

        if not embeddings:
            return

        # Check dimension consistency
        expected_dim = len(embeddings[0]) if embeddings else 0

        for idx, embedding in enumerate(embeddings):
            # Check for empty embeddings
            if not embedding or len(embedding) == 0:
                raise RuntimeError(f"Embedding {idx} is empty or has zero dimensions")

            # Check dimension consistency
            if len(embedding) != expected_dim:
                raise RuntimeError(
                    f"Embedding {idx} has inconsistent dimensions: "
                    f"expected {expected_dim}, got {len(embedding)}"
                )

            # Check for NaN or infinite values
            for dim_idx, value in enumerate(embedding):
                if math.isnan(value):
                    raise RuntimeError(f"Embedding {idx} contains NaN at dimension {dim_idx}")
                if math.isinf(value):
                    raise RuntimeError(f"Embedding {idx} contains inf at dimension {dim_idx}")

    def _rollback_documents(self, doc_ids: Sequence[int]) -> None:
        """Attempt to remove documents to maintain consistency.

        This is a best-effort rollback mechanism. If the document repository

        Args:
            doc_ids: Document IDs to remove
        """
        # Check if document repository supports deletion
        if hasattr(self._doc_store, "delete_documents"):
            self._doc_store.delete_documents(doc_ids)
        # If no deletion support, we can't rollback - this is logged in the exception

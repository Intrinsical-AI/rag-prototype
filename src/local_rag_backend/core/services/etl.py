# src/core/etl.py
"""
ETL service for document ingestion, embedding, and storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from local_rag_backend.core.ports import (
        DocumentRepoPort,
        EmbedderPort,
        VectorRepoPort,
    )


class ETLService:
    """Orchestrates document ingestion, embedding, and storage."""

    def __init__(
        self,
        doc_storage: DocumentRepoPort,
        vec_storage: VectorRepoPort,
        embedder: EmbedderPort,
    ):
        self._doc_store = doc_storage
        self._vec_store = vec_storage
        self._embedder = embedder

    def ingest(self, texts: Sequence[str]) -> Sequence[int]:
        """
        Processes and stores a sequence of texts.

        Returns:
            A sequence of unique integer IDs for the stored documents.
        """
        if not texts:
            return []

        # Store documents and get their IDs
        doc_ids = self._doc_store.store_documents(texts)

        # Generate and store vector embeddings
        embeddings = self._embedder.embed(texts)

        # Upsert embeddings into vector store
        self._vec_store.upsert(doc_ids, embeddings)

        return doc_ids

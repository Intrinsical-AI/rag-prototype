# src/core/etl.py
"""
ETL service for document ingestion, embedding, and storage.
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from local_rag_backend.core.ports import DocumentRepoPort, EmbedderPort, VectorRepoPort


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

        # 1) Compute embeddings first: avoids storing documents when embedding fails.
        embeddings = self._embedder.embed(texts)
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Embedder returned {len(embeddings)} embeddings for {len(texts)} texts."
            )

        # 2) Store documents and get their IDs
        doc_ids = list(self._doc_store.store_documents(texts))
        if len(doc_ids) != len(texts):
            raise ValueError(f"Doc repo returned {len(doc_ids)} ids for {len(texts)} texts.")

        # 3) Upsert embeddings into vector store. If it fails, best-effort rollback SQL inserts.
        try:
            self._vec_store.upsert(doc_ids, embeddings)
        except Exception:
            # Best-effort: if the vector store partially wrote state (or mutated in-memory),
            # attempt to remove those IDs too.
            with suppress(Exception):
                self._vec_store.delete(doc_ids)
            with suppress(Exception):
                self._doc_store.delete_documents(doc_ids)
            raise

        return doc_ids

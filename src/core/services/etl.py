# src/core/etl.py
from __future__ import annotations

from typing import Sequence

from src.core.ports import DocumentRepoPort, EmbedderPort, VectorRepoPort


class ETLService:
    """
    Encapsula la lógica de ingestión:
      1) almacenar docs en SQL
      2) generar embeddings
      3) almacenar embeddings en FAISS (u otro motor vectorial)
    """

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
        # 1) SQL
        ids = self._doc_store.store_documents(texts)

        # 2) Embeddings y vector store
        embeddings = self._embedder.embed(texts)
        self._vec_store.upsert(ids, embeddings)

        return ids

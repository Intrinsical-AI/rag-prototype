"""
Intrinsical-AI RAG Prototype
Copyright (c) 2025 Intrinsical-AI

Module: Dependency Injection
Purpose: FastAPI dependency injection for RAG service components.
         Implements singleton pattern and configurable service composition.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from local_rag_backend.core.services.rag import RagService
from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
    SentenceTransformerEmbedder,
)
from local_rag_backend.infrastructure.llms.ollama_chat import OllamaGenerator
from local_rag_backend.infrastructure.llms.openai_chat import OpenAIGenerator
from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import (
    HistorySqlStorage,
    SqlDocumentStorage,
)
from local_rag_backend.infrastructure.retrieval.dense_faiss import DenseFaissRetriever
from local_rag_backend.infrastructure.retrieval.hybrid import HybridRetriever
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever
from local_rag_backend.settings import settings
from local_rag_backend.utils import get_corpus_and_ids

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from local_rag_backend.core.ports import (
        DocumentRepoPort,
        EmbedderPort,
        GeneratorPort,
        QAHistoryPort,
        RetrieverPort,
        VectorRepoPort,
    )


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    """Build and return a singleton RAG service instance.

    Creates the complete RAG service with all dependencies based on configuration.
    Uses singleton pattern to ensure consistent service instance across requests.

    Returns:
        Configured RagService instance ready for query processing

    Raises:
        RuntimeError: If no LLM provider is configured
    """
    logger.info(f"Creating RAG service with retrieval mode: '{settings.retrieval_mode}'")

    # --- Document Storage ---
    doc_repo: DocumentRepoPort = SqlDocumentStorage()

    # --- Retrieval Strategy ---
    retriever = _build_retriever(doc_repo)

    # --- LLM Generator ---
    generator = _build_generator()

    # --- History Storage ---
    history_repo: QAHistoryPort = HistorySqlStorage()

    return RagService(retriever=retriever, generator=generator, history_storage=history_repo)


def _build_retriever(doc_repo: DocumentRepoPort) -> RetrieverPort:
    """Build retriever based on configured retrieval mode."""
    mode = settings.retrieval_mode

    if mode == "sparse":
        return _build_sparse_retriever(doc_repo)
    elif mode == "dense":
        return _build_dense_retriever(doc_repo)
    elif mode == "hybrid":
        return _build_hybrid_retriever(doc_repo)
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")


def _build_sparse_retriever(doc_repo: DocumentRepoPort) -> SparseBM25Retriever:
    """Build BM25-based sparse retriever."""
    corpus, doc_ids = get_corpus_and_ids(doc_repo)
    return SparseBM25Retriever(documents=corpus, doc_ids=doc_ids, doc_repo=doc_repo)


def _build_dense_retriever(doc_repo: DocumentRepoPort) -> DenseFaissRetriever:
    """Build FAISS-based dense retriever."""
    embedder: EmbedderPort = SentenceTransformerEmbedder(model_name=settings.st_embedding_model)
    vector_repo: VectorRepoPort = FaissVectorStorage(
        index_path=settings.index_path, id_map_path=settings.id_map_path, dim=embedder.dim
    )
    return DenseFaissRetriever(embedder=embedder, faiss_index=vector_repo, doc_repo=doc_repo)


def _build_hybrid_retriever(doc_repo: DocumentRepoPort) -> HybridRetriever:
    """Build hybrid retriever combining dense and sparse approaches."""
    dense_retriever = _build_dense_retriever(doc_repo)
    sparse_retriever = _build_sparse_retriever(doc_repo)

    return HybridRetriever(
        dense=dense_retriever,
        sparse=sparse_retriever,
        alpha=settings.hybrid_retrieval_alpha,
    )


def _build_generator() -> GeneratorPort:
    """Build LLM generator based on configuration."""
    if settings.ollama_enabled:
        return OllamaGenerator()
    elif settings.openai_api_key:
        return OpenAIGenerator()
    else:
        raise RuntimeError("No LLM configured. Set OPENAI_API_KEY or enable OLLAMA_ENABLED.")

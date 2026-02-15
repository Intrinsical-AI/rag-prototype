"""
Composition root (hex architecture).

This module is the stable import path for wiring ports to adapters based on `settings`.
FastAPI dependencies re-export `get_rag_service()` for DI convenience.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from local_rag_backend.core.services.rag import RagService
from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder
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


def _build_embedder() -> EmbedderPort:
    """
    Choose an embedder for dense/hybrid retrieval.

    Preference order:
    1) OpenAI embeddings when `OPENAI_API_KEY` is configured (no heavy deps).
    2) SentenceTransformers when installed (requires `dense-st` extra).
    """
    if settings.openai_api_key:
        return OpenAIEmbedder()
    try:
        return SentenceTransformerEmbedder(model_name=settings.st_embedding_model)
    except RuntimeError as e:
        raise RuntimeError(
            "Dense/hybrid retrieval requires an embeddings backend. "
            "Either set OPENAI_API_KEY to use OpenAI embeddings, or install the "
            "'dense-st' extra for SentenceTransformers (e.g. `uv sync --extra dense-st`)."
        ) from e


def build_rag_service() -> RagService:
    """Build a RagService instance based on current settings (no caching)."""
    logger.info("Creating RAG service with retrieval mode: '%s'", settings.retrieval_mode)

    # 1. Persistence Ports
    doc_repo: DocumentRepoPort = SqlDocumentStorage()

    # 2. Retriever Port
    if settings.retrieval_mode == "sparse":
        corpus, doc_ids = get_corpus_and_ids(doc_repo)
        retriever: RetrieverPort = SparseBM25Retriever(
            documents=corpus, doc_ids=doc_ids, doc_repo=doc_repo
        )
    else:
        embedder: EmbedderPort = _build_embedder()

        vector_repo: VectorRepoPort = FaissVectorStorage(
            index_path=settings.index_path, id_map_path=settings.id_map_path, dim=embedder.dim
        )
        dense_retriever = DenseFaissRetriever(
            embedder=embedder, faiss_index=vector_repo, doc_repo=doc_repo
        )
        if settings.retrieval_mode == "dense":
            retriever = dense_retriever
        else:  # hybrid
            corpus, doc_ids = get_corpus_and_ids(doc_repo)
            sparse_retriever = SparseBM25Retriever(
                documents=corpus, doc_ids=doc_ids, doc_repo=doc_repo
            )
            retriever = HybridRetriever(
                dense=dense_retriever,
                sparse=sparse_retriever,
                alpha=settings.hybrid_retrieval_alpha,
            )

    # 3. Generator Port
    generator: GeneratorPort
    if settings.ollama_enabled:
        generator = OllamaGenerator()
    elif settings.openai_api_key:
        generator = OpenAIGenerator()
    else:
        raise RuntimeError("No LLM configured. Set OPENAI_API_KEY or enable OLLAMA_ENABLED.")

    # 4. History Storage
    history_repo: QAHistoryPort = HistorySqlStorage()
    return RagService(retriever=retriever, generator=generator, history_storage=history_repo)


@lru_cache(maxsize=1)
def _get_cached_rag_service() -> RagService:
    return build_rag_service()


async def get_rag_service() -> RagService:
    """FastAPI dependency wrapper (async to avoid anyio threadpool for sync callables)."""
    return _get_cached_rag_service()


def reset_rag_service() -> None:
    """Clear the cached singleton (useful for tests)."""
    _get_cached_rag_service.cache_clear()


__all__ = ["build_rag_service", "get_rag_service", "reset_rag_service"]

"""
Composition root (hex architecture).

This module is the stable import path for wiring ports to adapters based on `settings`.
FastAPI dependencies re-export `get_rag_service()` for DI convenience.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from local_rag_backend.app.composition import (
    build_generator_from_settings,
    build_retriever_with_default_embedder_from_settings,
    resolve_preferred_llm_provider,
)
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
    SystemStateStorage,
)
from local_rag_backend.infrastructure.retrieval.dense_faiss import DenseFaissRetriever
from local_rag_backend.infrastructure.retrieval.hybrid import HybridRetriever
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever
from local_rag_backend.settings import settings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from local_rag_backend.core.ports import (
        DocumentRepoPort,
        GeneratorPort,
        QAHistoryPort,
        RetrieverPort,
    )

_RAG_SERVICE_STATE_KEY = "rag_service"
_system_state = SystemStateStorage()


def build_rag_service() -> RagService:
    """Build a RagService instance based on current settings (no caching)."""
    logger.info("Creating RAG service with retrieval mode: '%s'", settings.retrieval_mode)

    # 1. Persistence Ports
    doc_repo: DocumentRepoPort = SqlDocumentStorage()

    # 2. Retriever Port
    retriever: RetrieverPort = build_retriever_with_default_embedder_from_settings(
        settings_obj=settings,
        retrieval_mode=settings.retrieval_mode,
        doc_repo=doc_repo,
        openai_embedder_factory=OpenAIEmbedder,
        st_embedder_factory=lambda model_name: SentenceTransformerEmbedder(model_name=model_name),
        sparse_retriever_factory=SparseBM25Retriever,
        dense_retriever_factory=DenseFaissRetriever,
        hybrid_retriever_factory=HybridRetriever,
        vector_repo_factory=FaissVectorStorage,
    )

    # 3. Generator Port
    generator: GeneratorPort
    preferred_provider = resolve_preferred_llm_provider(settings_obj=settings)
    generator = build_generator_from_settings(
        settings_obj=settings,
        llm_provider=preferred_provider,
        openai_generator_factory=OpenAIGenerator,
        ollama_generator_factory=OllamaGenerator,
    )

    # 4. History Storage
    history_repo: QAHistoryPort = HistorySqlStorage()
    return RagService(retriever=retriever, generator=generator, history_storage=history_repo)


def _read_rag_service_version() -> int:
    try:
        return _system_state.get_version(_RAG_SERVICE_STATE_KEY)
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to read RAG service version from system_state: %s", e)
        return 0


@lru_cache(maxsize=1)
def _get_cached_rag_service(_version: int) -> RagService:
    return build_rag_service()


async def get_rag_service() -> RagService:
    """FastAPI dependency wrapper (async to avoid anyio threadpool for sync callables)."""
    # Multi-process invalidation: version is shared in SQLite system_state.
    return _get_cached_rag_service(_read_rag_service_version())


def reset_rag_service() -> None:
    """Clear the cached singleton (useful for tests)."""
    try:
        _system_state.bump_version(_RAG_SERVICE_STATE_KEY)
    except Exception as e:  # pragma: no cover
        # Don't fail request handlers/tests just because cache version couldn't be persisted.
        logger.warning("Failed to bump RAG service version in system_state: %s", e)
    _get_cached_rag_service.cache_clear()


__all__ = ["build_rag_service", "get_rag_service", "reset_rag_service"]

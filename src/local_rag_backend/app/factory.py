"""
Composition root (hex architecture).

This module is the stable import path for wiring ports to adapters based on `settings`.
FastAPI dependencies re-export `get_rag_service()` for DI convenience.
"""

from __future__ import annotations

import logging
import os
import tempfile
from functools import lru_cache
from time import time_ns
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
)
from local_rag_backend.infrastructure.retrieval.dense_faiss import DenseFaissRetriever
from local_rag_backend.infrastructure.retrieval.hybrid import HybridRetriever
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever
from local_rag_backend.settings import settings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from local_rag_backend.core.ports import (
        DocumentRepoPort,
        GeneratorPort,
        QAHistoryPort,
        RetrieverPort,
    )

_RELOAD_TOKEN_FILENAME = ".rag_service_reload_token"  # noqa: S105


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


def _reload_token_path() -> Path:
    # Keep it in the shared coordination dir so multi-worker/CLI processes stay in sync.
    return settings.get_coordination_dir() / _RELOAD_TOKEN_FILENAME


def _read_reload_token() -> str:
    p = _reload_token_path()
    try:
        return p.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to read RAG reload token at %s: %s", p, e)
        return ""


def _write_reload_token(token: str) -> None:
    p = _reload_token_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=p.name + ".",
        suffix=".tmp",
        dir=p.parent,
        delete=False,
    ) as tmp:
        tmp.write(token)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, p)


@lru_cache(maxsize=1)
def _get_cached_rag_service(_reload_token: str) -> RagService:
    return build_rag_service()


async def get_rag_service() -> RagService:
    """FastAPI dependency wrapper (async to avoid anyio threadpool for sync callables)."""
    # Multi-worker invalidation: other processes can "bust" the cache by updating the token file.
    return _get_cached_rag_service(_read_reload_token())


def reset_rag_service() -> None:
    """Clear the cached singleton (useful for tests)."""
    try:
        _write_reload_token(str(time_ns()))
    except Exception as e:  # pragma: no cover
        # Don't fail request handlers/tests just because the cache token couldn't be persisted.
        logger.warning("Failed to write RAG reload token: %s", e)
    _get_cached_rag_service.cache_clear()


__all__ = ["build_rag_service", "get_rag_service", "reset_rag_service"]

"""Factory/DI entrypoints backed by the centralized app container."""

from __future__ import annotations

import logging
from threading import Lock
from typing import TYPE_CHECKING, Any

from local_rag_backend.app import container as app_container_module
from local_rag_backend.app.app_context import AppContext
from local_rag_backend.app.container import AppContainer
from local_rag_backend.core.services.dense_upsert import (
    precompute_vectors_for_changed_items,
    sync_dense_after_upsert,
)
from local_rag_backend.core.services.maintenance import (
    delete_documents_multi_store,
    delete_external_ids_multi_store,
    rebuild_index_from_db,
)
from local_rag_backend.core.services.rag import RagService
from local_rag_backend.core.services.reranking import RerankingRetriever
from local_rag_backend.core.services.write_lock import multi_store_write_lock
from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder
from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
    SentenceTransformerEmbedder,
)
from local_rag_backend.infrastructure.llms.ollama_chat import OllamaGenerator
from local_rag_backend.infrastructure.llms.openai_chat import OpenAIGenerator
from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
from local_rag_backend.infrastructure.persistence.faiss.manifest import purge_index_artifacts
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import (
    HistorySqlStorage,
    SqlDocumentStorage,
    SystemStateStorage,
)
from local_rag_backend.infrastructure.retrieval.dense_faiss import DenseFaissRetriever
from local_rag_backend.infrastructure.retrieval.hybrid import HybridRetriever
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever
from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_RAG_SERVICE_STATE_KEY = AppContainer.RAG_SERVICE_STATE_KEY
_APP_CONTEXT: AppContext | None = None
_APP_CONTEXT_LOCK = Lock()


def _build_container(
    *,
    system_state_factory: Callable[[], SystemStateStorage] | None = None,
) -> AppContainer:
    resolved_system_state_factory = system_state_factory or SystemStateStorage

    # Keep AppContainer as the default composition source.
    # We only inject explicit overrides when tests monkeypatch factory-level seams.
    overrides: dict[str, Any] = {}
    if OpenAIEmbedder is not app_container_module.OpenAIEmbedder:
        overrides["openai_embedder_factory"] = OpenAIEmbedder
    if SentenceTransformerEmbedder is not app_container_module.SentenceTransformerEmbedder:
        overrides["st_embedder_factory"] = lambda model_name: SentenceTransformerEmbedder(
            model_name=model_name
        )
    if OpenAIGenerator is not app_container_module.OpenAIGenerator:
        overrides["openai_generator_factory"] = OpenAIGenerator
    if OllamaGenerator is not app_container_module.OllamaGenerator:
        overrides["ollama_generator_factory"] = OllamaGenerator
    if SqlDocumentStorage is not app_container_module.SqlDocumentStorage:
        overrides["doc_repo_factory"] = SqlDocumentStorage
        overrides["build_upsert_doc"] = getattr(SqlDocumentStorage, "UpsertDoc", None)
    if HistorySqlStorage is not app_container_module.HistorySqlStorage:
        overrides["history_repo_factory"] = HistorySqlStorage
    if SparseBM25Retriever is not app_container_module.SparseBM25Retriever:
        overrides["sparse_retriever_factory"] = SparseBM25Retriever
    if DenseFaissRetriever is not app_container_module.DenseFaissRetriever:
        overrides["dense_retriever_factory"] = DenseFaissRetriever
    if HybridRetriever is not app_container_module.HybridRetriever:
        overrides["hybrid_retriever_factory"] = HybridRetriever
    if FaissVectorStorage is not app_container_module.FaissVectorStorage:
        overrides["vector_repo_factory"] = FaissVectorStorage
    if RerankingRetriever is not app_container_module.RerankingRetriever:
        overrides["reranker_factory"] = RerankingRetriever
    if (
        precompute_vectors_for_changed_items
        is not app_container_module.precompute_vectors_for_changed_items
    ):
        overrides["precompute_vectors_fn"] = precompute_vectors_for_changed_items
    if sync_dense_after_upsert is not app_container_module.sync_dense_after_upsert:
        overrides["sync_dense_fn"] = sync_dense_after_upsert
    if rebuild_index_from_db is not app_container_module.rebuild_index_from_db:
        overrides["rebuild_fn"] = rebuild_index_from_db
    if delete_documents_multi_store is not app_container_module.delete_documents_multi_store:
        overrides["delete_docs_fn"] = delete_documents_multi_store
    if delete_external_ids_multi_store is not app_container_module.delete_external_ids_multi_store:
        overrides["delete_external_ids_fn"] = delete_external_ids_multi_store
    if purge_index_artifacts is not app_container_module.purge_index_artifacts:
        overrides["purge_index_artifacts_fn"] = purge_index_artifacts
    if multi_store_write_lock is not app_container_module.multi_store_write_lock:
        overrides["write_lock"] = multi_store_write_lock
    if RagService is not app_container_module.RagService:
        overrides["rag_service_factory"] = RagService

    return AppContainer(
        settings_obj=settings,
        system_state_factory=resolved_system_state_factory,
        **overrides,
    )


def _build_app_context(
    *,
    system_state_factory: Callable[[], SystemStateStorage] | None = None,
) -> AppContext:
    container = _build_container(system_state_factory=system_state_factory)
    return AppContext(settings_obj=settings, container=container)


def get_app_context() -> AppContext:
    global _APP_CONTEXT
    if _APP_CONTEXT is not None:
        return _APP_CONTEXT
    with _APP_CONTEXT_LOCK:
        if _APP_CONTEXT is None:
            _APP_CONTEXT = _build_app_context()
        if _APP_CONTEXT is None:
            raise RuntimeError("AppContext failed to initialize")
        return _APP_CONTEXT


def reset_app_context() -> None:
    global _APP_CONTEXT
    with _APP_CONTEXT_LOCK:
        _APP_CONTEXT = None


def build_rag_service() -> RagService:
    """Build a RagService instance based on current settings (no caching)."""
    ctx = get_app_context()
    logger.info("Creating RAG service with retrieval mode: '%s'", ctx.settings.retrieval_mode)
    return ctx.container.build_rag_service()


async def get_rag_service() -> RagService:
    """Return cached RagService from the app container."""
    return get_app_context().container.get_rag_service()


def reset_rag_service() -> None:
    """Invalidate cached RagService across processes and refresh local app context."""
    if _APP_CONTEXT is None:
        _build_container().reset_rag_service()
        return
    get_app_context().container.reset_rag_service()
    reset_app_context()


__all__ = [
    "build_rag_service",
    "get_app_context",
    "get_rag_service",
    "reset_app_context",
    "reset_rag_service",
]

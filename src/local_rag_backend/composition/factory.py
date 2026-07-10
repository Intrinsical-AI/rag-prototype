"""Factory/DI entrypoints backed by the centralized app container."""

from __future__ import annotations

import logging
from threading import Lock
from typing import TYPE_CHECKING, Any

from local_rag_backend.composition.container import AppContainer
from local_rag_backend.composition.context import AppContext
from local_rag_backend.composition.runtime import RuntimeSnapshot, build_runtime_snapshot
from local_rag_backend.core.services.rag_runtime import RagService
from local_rag_backend.infrastructure.persistence.sql import (
    SystemStateStorage,
)
from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_APP_CONTEXT: AppContext | None = None
_APP_CONTEXT_LOCK = Lock()
_RUNTIME_WIRING_DEFAULTS = AppContainer.runtime_wiring_defaults()

# Public wiring symbols intentionally exposed for tests that monkeypatch factory-level
# adapter constructors.
OpenAIEmbedder = _RUNTIME_WIRING_DEFAULTS["openai_embedder_factory"]
SentenceTransformerEmbedder = _RUNTIME_WIRING_DEFAULTS["st_embedder_factory"]
OpenAIGenerator = _RUNTIME_WIRING_DEFAULTS["openai_generator_factory"]
OllamaGenerator = _RUNTIME_WIRING_DEFAULTS["ollama_generator_factory"]
SqlDocumentStorage = _RUNTIME_WIRING_DEFAULTS["doc_repo_factory"]
HistorySqlStorage = _RUNTIME_WIRING_DEFAULTS["history_repo_factory"]
SparseBM25Retriever = _RUNTIME_WIRING_DEFAULTS["sparse_retriever_factory"]
DenseVectorRetriever = _RUNTIME_WIRING_DEFAULTS["dense_retriever_factory"]
HybridRetriever = _RUNTIME_WIRING_DEFAULTS["hybrid_retriever_factory"]
VectorStorage = _RUNTIME_WIRING_DEFAULTS["vector_repo_factory"]
RerankingRetriever = _RUNTIME_WIRING_DEFAULTS["reranker_factory"]
rebuild_index_from_db = _RUNTIME_WIRING_DEFAULTS["rebuild_fn"]
purge_index_artifacts = _RUNTIME_WIRING_DEFAULTS["purge_index_artifacts_fn"]
multi_store_write_lock = _RUNTIME_WIRING_DEFAULTS["write_lock"]


_FACTORY_WIRING_SYMBOLS: dict[str, str] = {
    "openai_embedder_factory": "OpenAIEmbedder",
    "st_embedder_factory": "SentenceTransformerEmbedder",
    "openai_generator_factory": "OpenAIGenerator",
    "ollama_generator_factory": "OllamaGenerator",
    "doc_repo_factory": "SqlDocumentStorage",
    "history_repo_factory": "HistorySqlStorage",
    "sparse_retriever_factory": "SparseBM25Retriever",
    "dense_retriever_factory": "DenseVectorRetriever",
    "hybrid_retriever_factory": "HybridRetriever",
    "vector_repo_factory": "VectorStorage",
    "reranker_factory": "RerankingRetriever",
    "rebuild_fn": "rebuild_index_from_db",
    "purge_index_artifacts_fn": "purge_index_artifacts",
    "write_lock": "multi_store_write_lock",
    "rag_service_factory": "RagService",
}


def _container_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, symbol_name in _FACTORY_WIRING_SYMBOLS.items():
        value = globals().get(symbol_name, _RUNTIME_WIRING_DEFAULTS[key])
        if key == "st_embedder_factory":
            overrides[key] = lambda model_name, _factory=value: _factory(model_name=model_name)
            continue
        overrides[key] = value

    doc_repo_factory = overrides.get(
        "doc_repo_factory", _RUNTIME_WIRING_DEFAULTS["doc_repo_factory"]
    )
    overrides["build_upsert_doc"] = getattr(doc_repo_factory, "UpsertDoc", None)
    return overrides


def _build_container(
    *,
    system_state_factory: Callable[[], SystemStateStorage] | None = None,
) -> AppContainer:
    resolved_system_state_factory = system_state_factory or SystemStateStorage
    overrides = _container_overrides()

    return AppContainer.from_settings(
        settings,
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
        ctx = _APP_CONTEXT
    if ctx is None:
        raise RuntimeError("AppContext failed to initialize")
    return ctx


def get_runtime_snapshot() -> RuntimeSnapshot:
    """Return a fresh, typed runtime snapshot derived from current Settings."""
    return build_runtime_snapshot(settings)


def reset_app_context() -> None:
    global _APP_CONTEXT
    with _APP_CONTEXT_LOCK:
        _APP_CONTEXT = None


def build_rag_service() -> RagService:
    """Build a RagService instance based on current settings (no caching)."""
    ctx = get_app_context()
    logger.info(
        "Creating RAG service with retrieval mode: '%s'",
        ctx.runtime_snapshot.retrieval_mode,
    )
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
    "DenseVectorRetriever",
    "HistorySqlStorage",
    "HybridRetriever",
    "OllamaGenerator",
    "OpenAIEmbedder",
    "OpenAIGenerator",
    "RerankingRetriever",
    "SentenceTransformerEmbedder",
    "SparseBM25Retriever",
    "SqlDocumentStorage",
    "VectorStorage",
    "build_rag_service",
    "get_app_context",
    "get_rag_service",
    "get_runtime_snapshot",
    "multi_store_write_lock",
    "purge_index_artifacts",
    "rebuild_index_from_db",
    "reset_app_context",
    "reset_rag_service",
]

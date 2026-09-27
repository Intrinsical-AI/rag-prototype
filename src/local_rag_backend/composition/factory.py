"""Factory/DI entrypoints backed by the centralized app container."""

from __future__ import annotations

import logging
from threading import Lock

from local_rag_backend.composition.container import AppContainer
from local_rag_backend.composition.context import AppContext
from local_rag_backend.composition.runtime import RuntimeSnapshot, build_runtime_snapshot
from local_rag_backend.core.services.rag_runtime import RagService
from local_rag_backend.settings import get_settings

logger = logging.getLogger(__name__)

_APP_CONTEXT: AppContext | None = None
_APP_CONTEXT_LOCK = Lock()


def _build_app_context() -> AppContext:
    settings_obj = get_settings()
    container = AppContainer.from_settings(settings_obj)
    return AppContext(settings_obj=settings_obj, container=container)


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
    return build_runtime_snapshot(get_app_context().settings)


def reset_app_context() -> None:
    global _APP_CONTEXT
    with _APP_CONTEXT_LOCK:
        if _APP_CONTEXT is not None:
            _APP_CONTEXT.container.close()
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
    container = get_app_context().container
    return await container.blocking_executor().run_blocking(container.get_rag_service)


def reset_rag_service() -> None:
    """Invalidate cached RagService while retaining the container's owned resources."""
    get_app_context().container.reset_rag_service()


__all__ = [
    "build_rag_service",
    "get_app_context",
    "get_rag_service",
    "get_runtime_snapshot",
    "reset_app_context",
    "reset_rag_service",
]

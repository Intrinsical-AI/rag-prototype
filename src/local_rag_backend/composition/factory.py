"""Factory/DI entrypoints backed by the centralized app container."""

from __future__ import annotations

import logging

from local_rag_backend.composition.container import AppContainer
from local_rag_backend.composition.context import AppContext, ApplicationRuntime
from local_rag_backend.composition.runtime import RuntimeSnapshot, build_runtime_snapshot
from local_rag_backend.core.services.rag_runtime import RagService
from local_rag_backend.settings import get_settings

logger = logging.getLogger(__name__)


def _build_app_context() -> AppContext:
    settings_obj = get_settings()
    container = AppContainer.from_settings(settings_obj)
    return AppContext(settings_obj=settings_obj, container=container)


_PROCESS_RUNTIME = ApplicationRuntime(lambda: _build_app_context())


def get_app_context() -> AppContext:
    return _PROCESS_RUNTIME.context()


def get_runtime_snapshot() -> RuntimeSnapshot:
    """Return a fresh, typed runtime snapshot derived from current Settings."""
    return build_runtime_snapshot(get_app_context().settings)


def reset_app_context() -> None:
    _PROCESS_RUNTIME.close()


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

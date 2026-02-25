"""FastAPI dependency helpers (runtime context + composition root wrappers)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_backend.composition.factory import (
    get_app_context as _get_app_context,
    get_rag_service,
    reset_rag_service,
)
from local_rag_backend.infrastructure.persistence.sql.base import get_db

if TYPE_CHECKING:
    from local_rag_backend.composition.container import AppContainer
    from local_rag_backend.composition.context import AppContext
    from local_rag_backend.settings import Settings


def get_app_context() -> AppContext:
    return _get_app_context()


async def get_app_container_dependency() -> AppContainer:
    """Async DI shim to avoid FastAPI threadpool offload for lightweight reads."""
    return get_app_context().container


async def get_settings_dependency() -> Settings:
    """Async DI shim to avoid FastAPI threadpool offload for lightweight reads."""
    return get_app_context().settings


__all__ = [
    "get_app_container_dependency",
    "get_app_context",
    "get_db",
    "get_rag_service",
    "get_settings_dependency",
    "reset_rag_service",
]

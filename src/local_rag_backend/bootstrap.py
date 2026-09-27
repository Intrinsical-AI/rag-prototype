"""Public bootstrap helpers for using the project as a library."""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_backend.composition.factory import get_app_context

if TYPE_CHECKING:
    from local_rag_backend.core.services.rag_runtime import RagService


def bootstrap_rag_service() -> RagService:
    """
    Build a new RagService instance from current settings.

    Note: this is a non-cached builder. FastAPI uses an internal cached singleton via DI.
    """

    container = get_app_context().container
    container.initialize()
    return container.build_rag_service()


__all__ = ["bootstrap_rag_service"]

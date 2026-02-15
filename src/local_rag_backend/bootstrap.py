"""Public bootstrap helpers for using the project as a library."""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_backend.app.factory import build_rag_service
from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from local_rag_backend.core.services.rag import RagService


def bootstrap_rag_service() -> RagService:
    """
    Build a new RagService instance from current settings.

    Note: this is a non-cached builder. FastAPI uses an internal cached singleton via DI.
    """

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return build_rag_service()


__all__ = ["bootstrap_rag_service"]

"""FastAPI dependencies for the application."""

from src.app.factory import get_rag_service as _get_rag_service
from src.core.services.rag import RagService


def get_rag_service() -> RagService:
    """Return the singleton :class:`RagService` instance."""
    return _get_rag_service()


__all__ = ["get_rag_service"]

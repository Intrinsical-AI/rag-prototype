# src/app/dependencies.py

from local_rag_backend.app.factory import get_rag_service as _get_rag_service
from local_rag_backend.core.services.rag import RagService


def get_rag_service() -> RagService:
    """
    FastAPI dependency to get the RAG service instance.
    Uses the factory singleton pattern.
    """
    return _get_rag_service()

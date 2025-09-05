# src/app/dependencies.py

from src.app.factory import get_rag_service as _get_rag_service
from src.core.services.rag import RagService


def get_rag_service() -> RagService:
    """
    FastAPI dependency to get the RAG service instance.
    Uses the factory singleton pattern.
    """
    return _get_rag_service()

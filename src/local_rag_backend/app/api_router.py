# src/local_rag_backend/app/api_router.py

"""
FastAPI router for the application endpoints.
"""

from typing import Annotated, Any

import requests
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from local_rag_backend.app.dependencies import get_rag_service
from local_rag_backend.core.services.etl import ETLService
from local_rag_backend.core.services.rag import RagService
from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
    SentenceTransformerEmbedder,
)
from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
from local_rag_backend.infrastructure.persistence.sqlalchemy.base import engine as global_app_engine
from local_rag_backend.infrastructure.persistence.sqlalchemy.base import get_db
from local_rag_backend.infrastructure.persistence.sqlalchemy.crud import get_history
from local_rag_backend.infrastructure.persistence.sqlalchemy.models import (
    Document as DbDocument,
)
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import (
    SqlDocumentStorage,
)
from local_rag_backend.models import AskRequest, AskResponse, DocumentInDB, HistoryItem, QueryResult
from local_rag_backend.settings import settings

router = APIRouter()


# ---------------------- Health Check Endpoints ---------------------- #


@router.get("/health", tags=["Health"], summary="Health check endpoint")
def health_check() -> dict[str, str]:
    """
    Basic health check endpoint for Docker/K8s monitoring.
    Returns:
        dict: Simple health status
    """
    try:
        # Test database connectivity
        with global_app_engine.connect() as conn:
            conn.execute("SELECT 1")
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {e!s}")


@router.get("/ready", tags=["Health"], summary="Readiness check endpoint")
def readiness_check(service: RagService = Depends(get_rag_service)) -> dict[str, str]:
    """
    Readiness check endpoint for Docker/K8s monitoring.
    Verifies that the service is ready to handle requests.
    Returns:
        dict: Readiness status
    """
    try:
        # Test database connectivity
        with global_app_engine.connect() as conn:
            conn.execute("SELECT 1")

        # Test RAG service availability
        if service is None:
            raise HTTPException(status_code=500, detail="RAG service not initialized")

        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Readiness check failed: {e!s}")


@router.get("/health/ollama", tags=["Health"], summary="Ollama server health check")
def ollama_health_check() -> dict[str, Any]:
    """
    Checks if the Ollama server is running and accessible.

    Returns:
        dict: A dictionary with the status of the Ollama server.
    """
    try:
        response = requests.get(settings.ollama_base_url, timeout=5)
        response.raise_for_status()
        return {"status": "ok", "url": settings.ollama_base_url}
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama server not accessible at {settings.ollama_base_url}. Error: {e}",
        )


# ---------------------- API Endpoints ---------------------- #


@router.post("/ask", response_model=AskResponse, tags=["RAG"], summary="Ask a question using RAG")
def ask(request: AskRequest, service: RagService = Depends(get_rag_service)) -> AskResponse:
    """
    Ask a question using Retrieval-Augmented Generation.
    Args:
        request: Question and retrieval parameters
    Returns:
        AskResponse: Generated answer with source documents
    """
    rag_result = service.ask(question=request.question, top_k=request.k)
    docs = rag_result["docs"]
    scores = rag_result["scores"]

    sources = [
        QueryResult(
            document=DocumentInDB(id=doc.id, content=doc.content),  # extiende aquí si hay metadata
            score=score,
        )
        for doc, score in zip(docs, scores, strict=False)
    ]
    return AskResponse(answer=rag_result["answer"], sources=sources)


@router.get("/history", response_model=list[HistoryItem], tags=["RAG"], summary="Get query history")
def history(
    limit: int = Query(10, ge=1, le=100, description="Max number of history items to retrieve"),
    offset: int = Query(0, ge=0, description="Number of items to skip (useful for pagination)"),
    db: Session = Depends(get_db),
) -> list[HistoryItem]:
    """
    Retrieves historical Q&A pairs from the database.
    Args:
        limit (int): Maximum number of history records to return.
        offset (int): Number of records to skip.
        db (Session): Database session dependency.
    Returns:
        List[HistoryItem]: List of historical Q&A entries.
    """
    history_entries = get_history(db=db, limit=limit, offset=offset)

    return [
        HistoryItem(
            id=entry.id,
            question=entry.question,
            answer=entry.answer,
            created_at=(
                entry.created_at.isoformat()
                if hasattr(entry.created_at, "isoformat")
                else str(entry.created_at)
            ),
            source_ids=entry.source_ids or [],
        )
        for entry in history_entries
    ]


# --- DOCS API (minimal KB) ---


@router.get("/docs", response_model=list[DocumentInDB])
def list_docs(
    limit: int = Query(100, ge=1, le=1000, description="Max number of docs"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
) -> list[DocumentInDB]:
    docs = db.query(DbDocument).order_by(DbDocument.id.asc()).offset(offset).limit(limit).all()
    return [DocumentInDB.model_validate(d) for d in docs]


class IngestRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Raw texts to ingest")


class IngestResponse(BaseModel):
    count: int
    ids: list[int]


@router.post("/docs", response_model=IngestResponse)
def ingest_docs(payload: Annotated[IngestRequest, Body(...)]) -> IngestResponse:
    texts = [t.strip() for t in payload.texts if t and t.strip()]
    if not texts:
        return IngestResponse(count=0, ids=[])

    # Repos
    doc_repo = SqlDocumentStorage()

    if settings.retrieval_mode in ("dense", "hybrid"):
        embedder = SentenceTransformerEmbedder(model_name=settings.st_embedding_model)
        vec = FaissVectorStorage(
            index_path=settings.index_path,
            id_map_path=settings.id_map_path,
            dim=embedder.dim,
        )
        etl = ETLService(doc_repo, vec, embedder)
        ids = list(etl.ingest(texts))
    else:
        ids = list(doc_repo.store_documents(texts))

    return IngestResponse(count=len(ids), ids=ids)

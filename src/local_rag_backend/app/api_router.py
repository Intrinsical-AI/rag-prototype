# src/app/api_router.py

"""
FastAPI router for the application endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from local_rag_backend.app.dependencies import get_rag_service
from local_rag_backend.core.services.rag import RagService
from local_rag_backend.infrastructure.persistence.sqlalchemy.base import engine as global_app_engine
from local_rag_backend.infrastructure.persistence.sqlalchemy.base import get_db
from local_rag_backend.infrastructure.persistence.sqlalchemy.crud import get_history
from local_rag_backend.models import AskRequest, AskResponse, DocumentInDB, HistoryItem, QueryResult

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

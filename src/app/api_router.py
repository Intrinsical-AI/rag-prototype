# src/app/api_router.py

"""
FastAPI router for the application endpoints.
"""

from typing import List

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.app.dependencies import get_rag_service
from src.core.services.rag import RagService
from src.infrastructure.persistence.sqlalchemy.base import get_db
from src.infrastructure.persistence.sqlalchemy.crud import get_history
from src.models import AskRequest, AskResponse, DocumentInDB, HistoryItem, QueryResult

router = APIRouter()


# ---------------------- API Endpoints ---------------------- #


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest, service: RagService = Depends(get_rag_service)
) -> AskResponse:
    rag_result = service.ask(question=request.question, top_k=request.k)
    docs = rag_result["docs"]
    scores = rag_result["scores"]

    sources = [
        QueryResult(
            document=DocumentInDB(
                id=doc.id, content=doc.content
            ),  # extiende aquí si hay metadata
            score=score,
        )
        for doc, score in zip(docs, scores)
    ]
    return AskResponse(answer=rag_result["answer"], sources=sources)


@router.get("/history", response_model=List[HistoryItem])
def history(
    limit: int = Query(
        10, ge=1, le=100, description="Max number of history items to retrieve"
    ),
    offset: int = Query(
        0, ge=0, description="Number of items to skip (useful for pagination)"
    ),
    db: Session = Depends(get_db),
) -> List[HistoryItem]:
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
            created_at=entry.created_at.isoformat(),
            source_ids=entry.source_ids or [],
        )
        for entry in history_entries
    ]

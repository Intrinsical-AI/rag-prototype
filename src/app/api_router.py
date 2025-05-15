# src/app/api_router.py

"""
FastAPI router for the application endpoints.
"""

from typing import List

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.db.base import get_db
from src.core.rag import RagService
from src.app.dependencies import get_rag_service
from src.db import crud

router = APIRouter()


# ------------------- Pydantic Schemas ------------------- #

class AskRequest(BaseModel):
    """Request schema for the `/ask` endpoint."""
    question: str = Field(..., description="User's question")
    k: int = Field(3, ge=1, le=10, description="Number of documents to retrieve")


class AskResponse(BaseModel):
    """Response schema for the `/ask` endpoint."""
    answer: str
    source_ids: List[int]


class HistoryItem(BaseModel):
    """Schema representing a single history item."""
    id: int
    question: str
    answer: str
    created_at: str


# ---------------------- API Endpoints ---------------------- #

@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    db: Session = Depends(get_db),
    service: RagService = Depends(get_rag_service),
) -> AskResponse:
    """
    Handles a user's question and retrieves an AI-generated answer using the RAG service.

    Args:
        request (AskRequest): Request body containing the user's question and the k-value.
        db (Session): Database session dependency.
        service (RagService): RAG service dependency.

    Returns:
        AskResponse: Generated answer and source document IDs.
    """
    return service.ask(db=db, question=request.question, k=request.k)


@router.get("/history", response_model=List[HistoryItem])
def history(
    limit: int = Query(10, ge=1, le=100, description="Max number of history items to retrieve"),
    offset: int = Query(0, ge=0, description="Number of items to skip (useful for pagination)"),
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
    history_entries = crud.get_history(db=db, limit=limit, offset=offset)

    return [
        HistoryItem(
            id=entry.id,
            question=entry.question,
            answer=entry.answer,
            created_at=entry.created_at.isoformat(),
        )
        for entry in history_entries
    ]

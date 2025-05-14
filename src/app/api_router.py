"""
File: src/app/api_router.py
Path: src/app/api_router.py
FastAPI router for the application endpoints.
"""

from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.db.base import get_db
from src.core.rag import RagService
from src.app.dependencies import get_rag_service

router = APIRouter()


# ------------------ Pydantic Schemas ------------------ #
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    source_ids: List[int]


class HistoryItem(BaseModel):
    id: int
    question: str
    answer: str
    created_at: str


# ------------------ Endpoints ------------------ #
@router.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    db: Session = Depends(get_db),
    service: RagService = Depends(get_rag_service),
):
    return service.ask(db, req.question)

@router.get("/history", response_model=List[HistoryItem])
def history(
    limit: int = 10,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    from src.db import crud
    qa_history_orm_objects = crud.get_history(db, limit, offset)
    
    # Convert
    history_list_for_response = []
    for item_orm in qa_history_orm_objects:
        history_list_for_response.append(
            HistoryItem(
                id=item_orm.id,
                question=item_orm.question,
                answer=item_orm.answer,
                created_at=item_orm.created_at.isoformat() # explicit conversion
            )
        )
    return history_list_for_response

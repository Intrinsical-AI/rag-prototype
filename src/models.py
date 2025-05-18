# src/models.py
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentInDB(BaseModel):
    id: int
    content: str
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        orm_mode = True


class QueryResult(BaseModel):
    document: DocumentInDB
    score: float


# API
class AskRequest(BaseModel):
    """Request schema for the `/ask` endpoint."""

    question: str = Field(..., description="User's question")
    k: int = Field(3, ge=1, le=10, description="Number of documents to retrieve")


class AskResponse(BaseModel):
    answer: str
    sources: List[QueryResult]


class HistoryItem(BaseModel):
    id: int
    question: str
    answer: str
    created_at: str
    source_ids: List[int] = []

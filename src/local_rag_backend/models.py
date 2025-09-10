# src/models.py

from pydantic import BaseModel, ConfigDict, Field


class DocumentInDB(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    content: str


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
    sources: list[QueryResult]


class HistoryItem(BaseModel):
    id: int
    question: str
    answer: str
    created_at: str
    source_ids: list[int] = Field(default_factory=list)

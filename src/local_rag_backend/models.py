"""
RAG Prototype - Intrinsical-AI (c) 2025
Author: Pablo Pintor
License: MIT

Models for the application.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DocumentInDB(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    content: str


class QueryResult(BaseModel):
    document: DocumentInDB
    score: float


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


class AskEvalConfig(BaseModel):
    """Ephemeral RAG configuration for per-request evaluation."""

    retrieval_mode: str = Field(..., description="sparse|dense|hybrid")
    k: int = Field(3, ge=1, le=10, description="Number of documents to retrieve")
    hybrid_alpha: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Hybrid alpha (weight for sparse) in [0,1]"
    )

    # Optional generator overrides
    llm_provider: str | None = Field(
        default=None, description="Optional override for generator provider: 'openai'|'ollama'"
    )
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    prompt_template: str | None = None


class AskEvalRequest(BaseModel):
    question: str = Field(..., description="User's question")
    config: AskEvalConfig


class AskEvalResponse(AskResponse):
    latency_ms: int | None = Field(default=None, description="Server-side latency in ms")

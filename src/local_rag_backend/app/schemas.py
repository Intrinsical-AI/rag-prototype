"""
Pydantic schemas for the HTTP API surface.

These are intentionally kept in the app layer to avoid coupling infrastructure/core
to transport concerns.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentInDB(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    content: str


class QueryResult(BaseModel):
    document: DocumentInDB
    score: float


class AskRequest(BaseModel):
    """Request schema for the `/ask` endpoint."""

    question: str = Field(..., min_length=1, max_length=4096, description="User's question")
    k: int = Field(3, ge=1, le=10, description="Number of documents to retrieve")

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("question must not be blank")
        return v2


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

    llm_provider: str | None = Field(
        default=None, description="Optional override for generator provider: 'openai'|'ollama'"
    )
    model: str | None = Field(default=None, max_length=256)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    prompt_template: str | None = Field(default=None, max_length=20000)


class AskEvalRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4096, description="User's question")
    config: AskEvalConfig

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("question must not be blank")
        return v2


class AskEvalResponse(AskResponse):
    latency_ms: int | None = Field(default=None, description="Server-side latency in ms")


class DeleteDocsRequest(BaseModel):
    ids: list[int] = Field(..., min_length=1, max_length=1000)


class DeleteDocsResponse(BaseModel):
    deleted_sql: int
    deleted_index: int | None = None
    rebuilt_index: bool = False


class DeleteDocsByExternalIdRequest(BaseModel):
    external_ids: list[str] = Field(..., min_length=1, max_length=512)


class DeleteDocsByExternalIdResponse(BaseModel):
    deleted_sql: int
    deleted_index: int | None = None
    tombstoned: int = 0
    missing_external_ids: list[str] = Field(default_factory=list)
    rebuilt_index: bool = False


class RebuildIndexResponse(BaseModel):
    indexed: int


class UpsertDocItem(BaseModel):
    external_id: str = Field(..., min_length=1, max_length=512)
    content: str = Field(..., min_length=1, max_length=20000)
    source_id: str | None = Field(default=None, max_length=1024)
    metadata: dict[str, Any] | None = None

    @field_validator("external_id")
    @classmethod
    def _external_id_not_blank(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("external_id must not be blank")
        return v2

    @field_validator("content")
    @classmethod
    def _content_not_blank(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("content must not be blank")
        return v2


class UpsertDocsRequest(BaseModel):
    docs: list[UpsertDocItem] = Field(..., min_length=1, max_length=64)


class UpsertDocResult(BaseModel):
    external_id: str
    id: int
    action: str
    content_changed: bool


class UpsertDocsResponse(BaseModel):
    inserted: int
    updated: int
    unchanged: int
    rebuilt_index: bool = False
    results: list[UpsertDocResult]

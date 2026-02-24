"""RAG bounded-context transport schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from local_rag_backend.app.schemas.shared import DocumentInDB  # noqa: TC001


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
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
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

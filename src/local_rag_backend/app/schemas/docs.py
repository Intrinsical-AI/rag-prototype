"""Docs bounded-context transport schemas."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator


class IngestRequest(BaseModel):
    texts: list[Annotated[str, Field(max_length=20000)]] = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Raw texts to ingest (max 64 items, 20k chars each)",
    )


class IngestResponse(BaseModel):
    count: int
    ids: list[str]


class DeleteDocsRequest(BaseModel):
    ids: list[Annotated[str, Field(min_length=1, max_length=256)]] = Field(
        ..., min_length=1, max_length=1000
    )


class DeleteDocsResponse(BaseModel):
    deleted_sql: int
    deleted_index: int | None = None
    rebuilt_index: bool = False


class DeleteDocsByExternalIdRequest(BaseModel):
    external_ids: list[Annotated[str, Field(min_length=1, max_length=512)]] = Field(
        ..., min_length=1, max_length=512
    )

    @field_validator("external_ids")
    @classmethod
    def _normalize_external_ids(cls, v: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for ext in v:
            ext_s = ext.strip()
            if not ext_s or ext_s in seen:
                continue
            seen.add(ext_s)
            normalized.append(ext_s)
        if not normalized:
            raise ValueError("external_ids must contain at least one non-blank value")
        return normalized


class DeleteDocsByExternalIdResponse(BaseModel):
    deleted_sql: int
    deleted_index: int | None = None
    tombstoned: int = 0
    missing_external_ids: list[str] = Field(default_factory=list)
    rebuilt_index: bool = False


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
    id: str
    action: str
    content_changed: bool


class UpsertDocsResponse(BaseModel):
    inserted: int
    updated: int
    unchanged: int
    rebuilt_index: bool = False
    results: list[UpsertDocResult]


class ImportResponse(BaseModel):
    """Response for POST /api/docs/import."""

    count: int = Field(..., description="Number of documents imported")
    ids: list[str] = Field(default_factory=list, description="IDs of imported documents")
    format_detected: str = Field(
        ..., description="Detected format: chatgpt_export or gemini_export"
    )

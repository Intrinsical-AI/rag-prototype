"""Docs bounded-context transport schemas."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator, model_validator

from local_rag_backend.http.schemas.rag_api_models import RetrievalFilterModel


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


class UpsertDocItem(BaseModel):
    external_id: str = Field(..., min_length=1, max_length=512)
    content: str = Field(..., min_length=1, max_length=20000)
    source_id: str | None = Field(default=None, max_length=1024)
    scope: str | None = Field(default=None, min_length=1, max_length=512)
    snapshot_id: str | None = Field(default=None, min_length=1, max_length=512)
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


class UpsertDocResult(BaseModel):
    external_id: str
    id: str
    action: str
    content_changed: bool


class CanonicalImportResponse(BaseModel):
    scope: str
    snapshot_id: str
    replace_scope: bool
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted_sql: int = 0
    deleted_index: int | None = None
    deleted_external_ids: list[str] = Field(default_factory=list)
    results: list[UpsertDocResult] = Field(default_factory=list)


class DocsQueryRequest(BaseModel):
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    filters: list[RetrievalFilterModel] = Field(default_factory=list)


class ImportResponse(BaseModel):
    """Response for POST /api/docs/import."""

    count: int = Field(..., description="Number of documents imported")
    ids: list[str] = Field(default_factory=list, description="IDs of imported documents")
    format_detected: str = Field(
        ..., description="Detected format: chatgpt_export or gemini_export"
    )


class DocsMutateRequest(BaseModel):
    op_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Optional idempotency key for mutation replay safety.",
    )
    upserts: list[UpsertDocItem] = Field(default_factory=list, max_length=256)
    delete_ids: list[Annotated[str, Field(min_length=1, max_length=256)]] = Field(
        default_factory=list,
        max_length=2048,
    )
    delete_external_ids: list[Annotated[str, Field(min_length=1, max_length=512)]] = Field(
        default_factory=list,
        max_length=2048,
    )

    @field_validator("delete_ids")
    @classmethod
    def _normalize_delete_ids(cls, v: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for doc_id in v:
            doc_id_s = doc_id.strip()
            if not doc_id_s or doc_id_s in seen:
                continue
            seen.add(doc_id_s)
            normalized.append(doc_id_s)
        return normalized

    @field_validator("delete_external_ids")
    @classmethod
    def _normalize_delete_external_ids(cls, v: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for ext in v:
            ext_s = ext.strip()
            if not ext_s or ext_s in seen:
                continue
            seen.add(ext_s)
            normalized.append(ext_s)
        return normalized

    @model_validator(mode="after")
    def _validate_payload(self) -> DocsMutateRequest:
        if not self.upserts and not self.delete_ids and not self.delete_external_ids:
            raise ValueError(
                "docs/mutate requires at least one operation: upserts, delete_ids, or delete_external_ids"
            )
        upsert_ext_ids = {str(doc.external_id).strip() for doc in self.upserts}
        conflict = upsert_ext_ids & set(self.delete_external_ids)
        if conflict:
            raise ValueError(
                "upserts and delete_external_ids cannot target the same external_id values"
            )
        return self


class DocsMutateResponse(BaseModel):
    op_id: str
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted_sql: int = 0
    deleted_index: int | None = None
    tombstoned: int = 0
    missing_external_ids: list[str] = Field(default_factory=list)
    index_rebuilt: bool = False
    index_doc_count: int | None = None
    results: list[UpsertDocResult] = Field(default_factory=list)

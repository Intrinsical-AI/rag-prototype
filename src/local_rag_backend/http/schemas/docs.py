"""Docs bounded-context transport schemas."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from local_rag_backend.core.services.docs_mutation_transport import (
    DocsMutationPayload,
    MutationUpsertPayload,
)
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


class UpsertDocItem(MutationUpsertPayload):
    """Transport alias for docs mutation upsert items."""


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


class DocsMutateRequest(DocsMutationPayload):
    """HTTP request alias using the shared transport validation contract."""


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

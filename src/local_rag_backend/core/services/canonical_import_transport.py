"""Shared transport validation/assembly for canonical import payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field, StrictBool, field_validator, model_validator

from local_rag_backend.core.services.external_canonical import normalize_external_canonical_payload
from local_rag_backend.core.use_cases.docs_import_canonical import (
    CanonicalImportDocumentInput,
    CanonicalImportRequestInput,
)


class CanonicalImportDocumentPayload(BaseModel):
    external_id: str = Field(..., min_length=1, max_length=512)
    content: str = Field(..., min_length=1, max_length=20000)
    source_id: str | None = Field(default=None, max_length=1024)
    metadata: dict[str, Any] | None = None

    @field_validator("external_id")
    @classmethod
    def _external_id_not_blank(cls, value: str) -> str:
        value2 = value.strip()
        if not value2:
            raise ValueError("external_id must not be blank")
        return value2

    @field_validator("content")
    @classmethod
    def _content_not_blank(cls, value: str) -> str:
        value2 = value.strip()
        if not value2:
            raise ValueError("content must not be blank")
        return value2


class CanonicalImportPayload(BaseModel):
    scope: str = Field(..., min_length=1, max_length=512)
    snapshot_id: str = Field(..., min_length=1, max_length=512)
    replace_scope: StrictBool = True
    documents: list[CanonicalImportDocumentPayload] = Field(
        default_factory=list, min_length=1, max_length=5000
    )

    @field_validator("scope", "snapshot_id")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value2 = value.strip()
        if not value2:
            raise ValueError("value must not be blank")
        return value2

    @model_validator(mode="after")
    def _validate_unique_external_ids(self) -> CanonicalImportPayload:
        seen: set[str] = set()
        duplicates: list[str] = []
        for document in self.documents:
            external_id = str(document.external_id).strip()
            if external_id in seen:
                duplicates.append(external_id)
            seen.add(external_id)
        if duplicates:
            raise ValueError(
                "documents.external_id values must be unique per canonical import: "
                + ", ".join(sorted(set(duplicates))[:10])
            )
        return self


def validate_canonical_import_payload(payload: Mapping[str, Any]) -> CanonicalImportPayload:
    """Validate and normalize a canonical import payload from any transport."""
    return CanonicalImportPayload.model_validate(normalize_external_canonical_payload(payload))


def resolve_canonical_import_replace_scope(
    payload: CanonicalImportPayload,
    *,
    replace_scope_override: bool | None = None,
) -> bool:
    if replace_scope_override is not None:
        return bool(replace_scope_override)
    return bool(payload.replace_scope)


def build_canonical_import_request_input(
    payload: CanonicalImportPayload,
    *,
    source: str,
    replace_scope_override: bool | None = None,
) -> CanonicalImportRequestInput:
    return CanonicalImportRequestInput(
        scope=payload.scope,
        snapshot_id=payload.snapshot_id,
        replace_scope=resolve_canonical_import_replace_scope(
            payload,
            replace_scope_override=replace_scope_override,
        ),
        documents=tuple(
            CanonicalImportDocumentInput(
                external_id=item.external_id,
                content=item.content,
                source_id=item.source_id,
                metadata=item.metadata,
            )
            for item in payload.documents
        ),
        source=source,
    )


def build_canonical_import_request_input_from_raw(
    payload: Mapping[str, Any],
    *,
    source: str,
    replace_scope_override: bool | None = None,
) -> CanonicalImportRequestInput:
    """Validate raw payload data and adapt it to the internal import request DTO."""
    return build_canonical_import_request_input(
        validate_canonical_import_payload(payload),
        source=source,
        replace_scope_override=replace_scope_override,
    )


__all__ = [
    "CanonicalImportDocumentPayload",
    "CanonicalImportPayload",
    "build_canonical_import_request_input",
    "build_canonical_import_request_input_from_raw",
    "resolve_canonical_import_replace_scope",
    "validate_canonical_import_payload",
]

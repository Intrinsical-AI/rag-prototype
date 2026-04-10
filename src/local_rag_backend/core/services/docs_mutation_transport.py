"""Shared transport validation/assembly for docs mutation payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from local_rag_backend.core.use_cases.docs_mutation import MutationIntent, MutationUpsertInput


class MutationUpsertPayload(BaseModel):
    external_id: str = Field(..., min_length=1, max_length=512)
    content: str = Field(..., min_length=1, max_length=20000)
    source_id: str | None = Field(default=None, max_length=1024)
    scope: str | None = Field(default=None, min_length=1, max_length=512)
    snapshot_id: str | None = Field(default=None, min_length=1, max_length=512)
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


class DocsMutationPayload(BaseModel):
    op_id: str | None = Field(default=None, min_length=1, max_length=128)
    upserts: list[MutationUpsertPayload] = Field(default_factory=list, max_length=256)
    delete_ids: list[str] = Field(default_factory=list, max_length=2048)
    delete_external_ids: list[str] = Field(default_factory=list, max_length=2048)

    @field_validator("delete_ids")
    @classmethod
    def _normalize_delete_ids(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for doc_id in values:
            doc_id_s = str(doc_id).strip()
            if not doc_id_s or doc_id_s in seen:
                continue
            seen.add(doc_id_s)
            normalized.append(doc_id_s)
        return normalized

    @field_validator("delete_external_ids")
    @classmethod
    def _normalize_delete_external_ids(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for external_id in values:
            ext_s = str(external_id).strip()
            if not ext_s or ext_s in seen:
                continue
            seen.add(ext_s)
            normalized.append(ext_s)
        return normalized

    @model_validator(mode="after")
    def _validate_payload(self) -> DocsMutationPayload:
        if not self.upserts and not self.delete_ids and not self.delete_external_ids:
            raise ValueError(
                "docs/mutate requires at least one operation: upserts, delete_ids, or delete_external_ids"
            )
        upsert_ext_ids = {str(item.external_id).strip() for item in self.upserts}
        conflict = upsert_ext_ids & set(self.delete_external_ids)
        if conflict:
            raise ValueError(
                "upserts and delete_external_ids cannot target the same external_id values"
            )
        return self


def validate_docs_mutation_payload(payload: Mapping[str, Any]) -> DocsMutationPayload:
    return DocsMutationPayload.model_validate(payload)


def build_docs_mutation_intent(
    payload: DocsMutationPayload,
    *,
    source: str,
) -> MutationIntent:
    return MutationIntent(
        op_id=str(payload.op_id or "").strip(),
        upserts=tuple(
            MutationUpsertInput(
                external_id=item.external_id,
                content=item.content,
                source_id=item.source_id,
                scope=item.scope,
                snapshot_id=item.snapshot_id,
                metadata=item.metadata,
            )
            for item in payload.upserts
        ),
        delete_ids=tuple(payload.delete_ids),
        delete_external_ids=tuple(payload.delete_external_ids),
        source=source,
    )


def build_docs_mutation_intent_from_raw(
    payload: Mapping[str, Any],
    *,
    source: str,
) -> MutationIntent:
    return build_docs_mutation_intent(validate_docs_mutation_payload(payload), source=source)


__all__ = [
    "DocsMutationPayload",
    "MutationUpsertPayload",
    "build_docs_mutation_intent",
    "build_docs_mutation_intent_from_raw",
    "validate_docs_mutation_payload",
]

"""App use-case result DTOs (transport-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UpsertDocResult:
    external_id: str
    id: str
    action: str
    content_changed: bool


@dataclass(frozen=True)
class MutationSummary:
    op_id: str
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted_sql: int = 0
    deleted_index: int | None = None
    tombstoned: int = 0
    missing_external_ids: list[str] | None = None
    index_rebuilt: bool = False
    index_doc_count: int | None = None
    results: list[UpsertDocResult] | None = None


@dataclass(frozen=True)
class CanonicalImportSummary:
    scope: str
    snapshot_id: str
    replace_scope: bool
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted_sql: int = 0
    deleted_index: int | None = None
    deleted_external_ids: list[str] | None = None
    results: list[UpsertDocResult] | None = None

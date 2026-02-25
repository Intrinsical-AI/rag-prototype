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
class UpsertDocsSummary:
    inserted: int
    updated: int
    unchanged: int
    rebuilt_index: bool
    results: list[UpsertDocResult]


@dataclass(frozen=True)
class DeleteDocsByExternalIdSummary:
    deleted_sql: int
    deleted_index: int | None
    tombstoned: int
    missing_external_ids: list[str]
    rebuilt_index: bool


@dataclass(frozen=True)
class DeleteDocsSummary:
    deleted_sql: int
    deleted_index: int | None
    rebuilt_index: bool


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

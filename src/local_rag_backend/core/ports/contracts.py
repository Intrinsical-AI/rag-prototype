"""
Application ports for docs/index mutation use cases.

These contracts isolate app-layer orchestration from concrete infra adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from contextlib import AbstractContextManager
    from pathlib import Path

    from local_rag_backend.core.domain.entities import Document
    from local_rag_backend.core.domain.profiles import StorageProfileRegistry
    from local_rag_backend.core.domain.types import DocId
    from local_rag_backend.core.ports import DocumentRepoPort, EmbedderPort, VectorRepoPort


class UpsertResultPort(Protocol):
    @property
    def external_id(self) -> str: ...

    @property
    def id(self) -> str: ...

    @property
    def action(self) -> str: ...

    @property
    def content_changed(self) -> bool: ...


class UpsertDocBuilderPort(Protocol):
    # Returns `object` (not a TypeVar) intentionally: the concrete builder produces an
    # ORM-specific row object that only the persistence adapter understands.  Making the
    # protocol generic would propagate a TypeVar through the entire mutation wiring for
    # no benefit at the app layer; DocsRepositoryPort.upsert_documents_by_external_id
    # accepts Sequence[object], which is the narrowest safe shared type.
    def __call__(
        self,
        *,
        external_id: str,
        content: str,
        source_id: str | None = None,
        scope: str | None = None,
        snapshot_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        chunk_dedup_sha256: str | None = None,
        embedding: Sequence[float] | None = None,
    ) -> object: ...


class DocsRepositoryPort(Protocol):
    """Contract for the docs repository used by mutation use cases.

    Required methods are called unconditionally by MutationCoordinator.
    Optional rollback/snapshot methods are guarded by hasattr in the coordinator
    and do not need to be implemented by every adapter.
    """

    # --- always required ---
    def get_tombstoned_external_ids(self, external_ids: Sequence[str]) -> set[str]: ...

    def upsert_documents_by_external_id(
        self, items: Sequence[object]
    ) -> tuple[list[UpsertResultPort], list[tuple[DocId, str]], list[DocId]]: ...

    def get(self, ids: Sequence[DocId]) -> Sequence[Document]: ...

    def delete_documents(self, ids: Sequence[DocId]) -> None: ...

    def delete_by_external_ids(
        self, external_ids: Sequence[str]
    ) -> tuple[int, list[DocId], list[str], int]: ...


MutationState = Literal[
    "PREPARED",
    "SQL_COMMITTED",
    "VECTOR_COMMITTED",
    "COMMITTED",
    "COMPENSATING",
    "ROLLED_BACK",
    "FAILED_NEEDS_RECOVERY",
]


@dataclass(frozen=True)
class MutationRecord:
    op_id: str
    state: MutationState
    intent: dict[str, Any]
    before_image: dict[str, Any] | None = None
    outcome: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    vector_attempted: bool = False

    def validate_outcome(self) -> None:
        """Reject incomplete commit evidence before replay or terminal filtering."""
        payload = self.outcome
        counters = {"inserted", "updated", "unchanged", "deleted_sql", "tombstoned"}
        nullable_counters = {"deleted_index", "index_doc_count"}
        keys = (
            counters
            | nullable_counters
            | {"op_id", "missing_external_ids", "index_rebuilt", "results"}
        )
        if not isinstance(payload, dict) or set(payload) != keys:
            raise ValueError("outcome is missing or incomplete")
        if payload["op_id"] != self.op_id:
            raise ValueError("outcome does not match op_id")
        for key in counters | nullable_counters:
            value = payload[key]
            if value is None and key in nullable_counters:
                continue
            if type(value) is not int or value < 0:
                raise ValueError(f"invalid outcome counter {key}")
        if type(payload["index_rebuilt"]) is not bool:
            raise ValueError("invalid outcome index_rebuilt")
        missing = payload["missing_external_ids"]
        if not isinstance(missing, list) or any(not isinstance(v, str) for v in missing):
            raise ValueError("invalid outcome missing_external_ids")
        results = payload["results"]
        if not isinstance(results, list):
            raise ValueError("invalid outcome results")
        for row in results:
            if not isinstance(row, dict) or set(row) != {
                "external_id",
                "id",
                "action",
                "content_changed",
            }:
                raise ValueError("invalid outcome result fields")
            if any(not isinstance(row[key], str) or not row[key] for key in ("external_id", "id")):
                raise ValueError("invalid outcome result identity")
            if row["action"] not in {"inserted", "updated", "unchanged"}:
                raise ValueError("invalid outcome action")
            if type(row["content_changed"]) is not bool:
                raise ValueError("invalid outcome content_changed")

    def validate_before_image(self) -> None:
        """A rollback must have complete snapshots, never default missing data to empty."""
        payload = self.before_image
        if not isinstance(payload, dict) or set(payload) != {"docs", "existing_tombstones"}:
            raise ValueError("before_image is missing or incomplete")
        snapshots, tombstones = payload["docs"], payload["existing_tombstones"]
        if not isinstance(tombstones, list) or any(
            not isinstance(value, str) or not value.strip() for value in tombstones
        ):
            raise ValueError("invalid before_image existing_tombstones")
        if not isinstance(snapshots, list):
            raise ValueError("invalid before_image docs")
        nullable_strings = {
            "external_id",
            "source_id",
            "scope",
            "snapshot_id",
            "content_sha256",
            "chunk_dedup_sha256",
        }
        for snapshot in snapshots:
            if not isinstance(snapshot, dict) or set(snapshot) != nullable_strings | {
                "id",
                "content",
                "metadata",
            }:
                raise ValueError("incomplete before_image document snapshot")
            for key in ("id", "content"):
                if not isinstance(snapshot[key], str) or not snapshot[key].strip():
                    raise ValueError(f"invalid before_image document {key}")
            for key in nullable_strings:
                if snapshot[key] is not None and not isinstance(snapshot[key], str):
                    raise ValueError(f"invalid before_image document {key}")
            if snapshot["metadata"] is not None and not isinstance(snapshot["metadata"], dict):
                raise ValueError("invalid before_image document metadata")


class MutationJournalPort(Protocol):
    def get(self, op_id: str) -> MutationRecord | None: ...
    def upsert(self, record: MutationRecord) -> None: ...
    def list_incomplete(self, *, limit: int = 100) -> list[MutationRecord]: ...


class WriteLockPort(Protocol):
    """Contract for the multi-store write lock callable."""

    def __call__(
        self,
        *,
        coordination_dir: Path | None = None,
        timeout_s: float | None = None,
        poll_s: float | None = None,
    ) -> AbstractContextManager[None]: ...


@dataclass(frozen=True)
class DocsMutationPorts:
    build_embedder: Callable[[], EmbedderPort]
    doc_repo_factory: Callable[[], DocsRepositoryPort]
    build_upsert_doc: UpsertDocBuilderPort
    vector_repo_factory: Callable[..., VectorRepoPort]
    reconcile_index: Callable[[], int]
    write_lock: WriteLockPort
    mutation_journal_factory: Callable[[], MutationJournalPort]
    storage_profile_registry: StorageProfileRegistry
    mutation_uow_factory: Callable[[], AbstractContextManager[None]] | None = None


@dataclass(frozen=True)
class IndexMutationPorts:
    build_embedder: Callable[[], EmbedderPort]
    doc_repo_factory: Callable[[], DocumentRepoPort]
    vector_repo_factory: Callable[..., VectorRepoPort]
    purge_index_artifacts_fn: Callable[..., None]
    rebuild_fn: Callable[..., int]

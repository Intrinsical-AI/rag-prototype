"""
Application ports for docs/index mutation use cases.

These contracts isolate app-layer orchestration from concrete infra adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from local_rag_backend.core.ports import EmbedderPort


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
    def __call__(
        self,
        *,
        external_id: str,
        content: str,
        source_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        chunk_dedup_sha256: str | None = None,
    ) -> Any: ...


class DocsRepositoryPort(Protocol):
    def get_tombstoned_external_ids(self, external_ids: Sequence[str]) -> set[str]: ...

    def upsert_documents_by_external_id(
        self, items: Sequence[Any]
    ) -> tuple[list[UpsertResultPort], list[tuple[str, str]], list[str]]: ...


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


class MutationJournalPort(Protocol):
    def get(self, op_id: str) -> MutationRecord | None: ...
    def upsert(self, record: MutationRecord) -> None: ...
    def delete(self, op_id: str) -> None: ...
    def list_incomplete(self, *, limit: int = 100) -> list[MutationRecord]: ...


@dataclass(frozen=True)
class DocsMutationPorts:
    build_embedder: Callable[[], EmbedderPort]
    doc_repo_factory: Callable[[], DocsRepositoryPort]
    build_upsert_doc: UpsertDocBuilderPort
    vector_repo_factory: Callable[..., Any]
    precompute_vectors_fn: Callable[..., dict[str, list[float]]]
    sync_dense_fn: Callable[..., bool]
    rebuild_fn: Callable[..., int]
    delete_docs_fn: Callable[..., tuple[int, int | None, bool]]
    delete_external_ids_fn: Callable[..., tuple[int, int | None, list[str], int, bool]]
    write_lock: Callable[..., Any]
    mutation_journal_factory: Callable[[], MutationJournalPort]
    storage_profile_registry: Any


@dataclass(frozen=True)
class IndexMutationPorts:
    build_embedder: Callable[[], EmbedderPort]
    doc_repo_factory: Callable[[], Any]
    vector_repo_factory: Callable[..., Any]
    purge_index_artifacts_fn: Callable[..., None]
    rebuild_fn: Callable[..., int]

"""
Shared builders for docs/index mutation ports.

These helpers let API and CLI reuse the same app-layer orchestration contracts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from local_rag_backend.app.application.storage_profiles import StorageProfileRegistry
from local_rag_backend.app.contracts.ports import (
    DocsMutationPorts,
    IndexMutationPorts,
)
from local_rag_backend.core.services.maintenance import (
    rebuild_index_from_db,
)
from local_rag_backend.core.services.write_lock import multi_store_write_lock
from local_rag_backend.infrastructure.persistence.shared.mutation_journal import FileMutationJournal
from local_rag_backend.infrastructure.persistence.vector.manifest import purge_index_artifacts
from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from local_rag_backend.core.ports import DocumentRepoPort, EmbedderPort, VectorRepoPort


def build_docs_mutation_ports(
    *,
    build_embedder: Callable[[], EmbedderPort],
    doc_repo_factory: Callable[[], Any] | None = None,
    build_upsert_doc: Any | None = None,
    vector_repo_factory: Callable[..., VectorRepoPort] | None = None,
    rebuild_fn: Callable[..., int] = rebuild_index_from_db,
    write_lock: Callable[..., Any] = multi_store_write_lock,
    mutation_journal_factory: Callable[..., Any] | None = None,
    storage_profile_registry: StorageProfileRegistry | None = None,
) -> DocsMutationPorts:
    if doc_repo_factory is None or build_upsert_doc is None:
        from local_rag_backend.infrastructure.persistence.sql.alchemy_engine import (
            SqlDocumentStorage,
        )

        repo_factory: Callable[[], Any] = doc_repo_factory or cast(
            "Callable[[], Any]", lambda: SqlDocumentStorage()
        )
        upsert_doc_builder = build_upsert_doc or SqlDocumentStorage.UpsertDoc
    else:
        repo_factory = doc_repo_factory
        upsert_doc_builder = build_upsert_doc

    if vector_repo_factory is None:
        from local_rag_backend.infrastructure.persistence.vector.storage import VectorStorage

        vec_factory: Callable[..., VectorRepoPort] = VectorStorage
    else:
        vec_factory = vector_repo_factory
    journal_factory = mutation_journal_factory or (
        lambda: FileMutationJournal(settings.get_coordination_dir() / ".mutation_journal")
    )
    profile_registry = storage_profile_registry or StorageProfileRegistry()
    return DocsMutationPorts(
        build_embedder=build_embedder,
        doc_repo_factory=repo_factory,
        build_upsert_doc=upsert_doc_builder,
        vector_repo_factory=vec_factory,
        rebuild_fn=rebuild_fn,
        write_lock=write_lock,
        mutation_journal_factory=journal_factory,
        storage_profile_registry=profile_registry,
    )


def build_index_mutation_ports(
    *,
    build_embedder: Callable[[], EmbedderPort],
    doc_repo_factory: Callable[[], DocumentRepoPort] | None = None,
    vector_repo_factory: Callable[..., VectorRepoPort] | None = None,
    purge_index_artifacts_fn: Callable[..., None] = purge_index_artifacts,
    rebuild_fn: Callable[..., int] = rebuild_index_from_db,
) -> IndexMutationPorts:
    if doc_repo_factory is None:
        from local_rag_backend.infrastructure.persistence.sql.alchemy_engine import (
            SqlDocumentStorage,
        )

        repo_factory: Callable[[], DocumentRepoPort] = cast(
            "Callable[[], DocumentRepoPort]", lambda: SqlDocumentStorage()
        )
    else:
        repo_factory = doc_repo_factory

    if vector_repo_factory is None:
        from local_rag_backend.infrastructure.persistence.vector.storage import VectorStorage

        vec_factory: Callable[..., VectorRepoPort] = VectorStorage
    else:
        vec_factory = vector_repo_factory
    return IndexMutationPorts(
        build_embedder=build_embedder,
        doc_repo_factory=repo_factory,
        vector_repo_factory=vec_factory,
        purge_index_artifacts_fn=purge_index_artifacts_fn,
        rebuild_fn=rebuild_fn,
    )

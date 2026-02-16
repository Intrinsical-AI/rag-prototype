"""
Shared builders for docs/index mutation ports.

These helpers let API and CLI reuse the same app-layer orchestration contracts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from local_rag_backend.app.services.ports import DocsMutationPorts, IndexMutationPorts
from local_rag_backend.core.services.dense_upsert import (
    precompute_vectors_for_changed_items,
    sync_dense_after_upsert,
)
from local_rag_backend.core.services.maintenance import (
    delete_documents_multi_store,
    delete_external_ids_multi_store,
    rebuild_index_from_db,
)
from local_rag_backend.infrastructure.persistence.faiss.manifest import purge_index_artifacts

if TYPE_CHECKING:
    from collections.abc import Callable

    from local_rag_backend.core.ports import EmbedderPort


def build_docs_mutation_ports(
    *,
    build_embedder: Callable[[], EmbedderPort],
    doc_repo_factory: Callable[[], Any] | None = None,
    build_upsert_doc: Any | None = None,
    vector_repo_factory: Callable[..., Any] | None = None,
    precompute_vectors_fn: Callable[..., dict[str, list[float]]] = precompute_vectors_for_changed_items,
    sync_dense_fn: Callable[..., bool] = sync_dense_after_upsert,
    rebuild_fn: Callable[..., int] = rebuild_index_from_db,
    delete_docs_fn: Callable[..., tuple[int, int | None, bool]] = delete_documents_multi_store,
    delete_external_ids_fn: Callable[..., tuple[int, int | None, list[str], int, bool]] = delete_external_ids_multi_store,
) -> DocsMutationPorts:
    if doc_repo_factory is None or build_upsert_doc is None:
        from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

        repo_factory: Callable[[], Any] = (
            doc_repo_factory or cast("Callable[[], Any]", lambda: SqlDocumentStorage())
        )
        upsert_doc_builder = build_upsert_doc or SqlDocumentStorage.UpsertDoc
    else:
        repo_factory = doc_repo_factory
        upsert_doc_builder = build_upsert_doc

    if vector_repo_factory is None:
        from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage

        vec_factory: Callable[..., Any] = FaissVectorStorage
    else:
        vec_factory = vector_repo_factory
    return DocsMutationPorts(
        build_embedder=build_embedder,
        doc_repo_factory=repo_factory,
        build_upsert_doc=upsert_doc_builder,
        vector_repo_factory=vec_factory,
        precompute_vectors_fn=precompute_vectors_fn,
        sync_dense_fn=sync_dense_fn,
        rebuild_fn=rebuild_fn,
        delete_docs_fn=delete_docs_fn,
        delete_external_ids_fn=delete_external_ids_fn,
    )


def build_index_mutation_ports(
    *,
    build_embedder: Callable[[], EmbedderPort],
    doc_repo_factory: Callable[[], Any] | None = None,
    vector_repo_factory: Callable[..., Any] | None = None,
    purge_index_artifacts_fn: Callable[..., None] = purge_index_artifacts,
    rebuild_fn: Callable[..., int] = rebuild_index_from_db,
) -> IndexMutationPorts:
    if doc_repo_factory is None:
        from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

        repo_factory: Callable[[], Any] = cast("Callable[[], Any]", lambda: SqlDocumentStorage())
    else:
        repo_factory = doc_repo_factory

    if vector_repo_factory is None:
        from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage

        vec_factory: Callable[..., Any] = FaissVectorStorage
    else:
        vec_factory = vector_repo_factory
    return IndexMutationPorts(
        build_embedder=build_embedder,
        doc_repo_factory=repo_factory,
        vector_repo_factory=vec_factory,
        purge_index_artifacts_fn=purge_index_artifacts_fn,
        rebuild_fn=rebuild_fn,
    )

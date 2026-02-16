"""
Application-layer index operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from local_rag_backend.core.services.maintenance import rebuild_index_from_db
from local_rag_backend.infrastructure.persistence.faiss.manifest import purge_index_artifacts

if TYPE_CHECKING:
    from collections.abc import Callable

    from local_rag_backend.core.ports import EmbedderPort
    from local_rag_backend.settings import Settings


def rebuild_index_sync(
    *,
    settings_obj: Settings,
    build_embedder: Callable[[], EmbedderPort],
    doc_repo_factory: Callable[[], Any],
    vector_repo_factory: Callable[..., Any],
    purge_index_artifacts_fn: Callable[..., None] = purge_index_artifacts,
    rebuild_fn: Callable[..., int] = rebuild_index_from_db,
) -> int:
    doc_repo = doc_repo_factory()
    embedder = build_embedder()

    purge_index_artifacts_fn(index_path=settings_obj.index_path, id_map_path=settings_obj.id_map_path)
    vec = vector_repo_factory(
        index_path=settings_obj.index_path,
        id_map_path=settings_obj.id_map_path,
        dim=embedder.dim,
    )
    return rebuild_fn(doc_repo=doc_repo, vec_repo=vec, embedder=embedder)

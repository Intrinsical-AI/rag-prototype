"""Application-layer index operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_rag_backend.app.contracts.ports import IndexMutationPorts
    from local_rag_backend.settings import Settings


def rebuild_index_sync(
    *,
    settings_obj: Settings,
    ports: IndexMutationPorts,
) -> int:
    doc_repo = ports.doc_repo_factory()
    embedder = ports.build_embedder()

    ports.purge_index_artifacts_fn(
        index_path=settings_obj.index_path, id_map_path=settings_obj.id_map_path
    )
    vec = ports.vector_repo_factory(
        index_path=settings_obj.index_path,
        id_map_path=settings_obj.id_map_path,
        dim=embedder.dim,
        backend=getattr(settings_obj, "vector_backend", "auto"),
    )
    return ports.rebuild_fn(doc_repo=doc_repo, vec_repo=vec, embedder=embedder)

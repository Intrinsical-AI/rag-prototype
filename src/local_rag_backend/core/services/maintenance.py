"""
Maintenance operations that must keep SQL + vector index consistent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Protocol

    from local_rag_backend.core.ports import DocumentRepoPort, EmbedderPort, VectorRepoPort

    class ExternalIdDeleteRepoPort(DocumentRepoPort, Protocol):
        def delete_by_external_ids(
            self, external_ids: Sequence[str]
        ) -> tuple[int, list[int], list[str], int]: ...


def rebuild_index_from_db(
    *,
    doc_repo: DocumentRepoPort,
    vec_repo: VectorRepoPort,
    embedder: EmbedderPort,
    batch_size: int = 128,
) -> int:
    docs = list(doc_repo.get_all_documents())
    if not docs:
        vec_repo.rebuild([], [])
        return 0

    ids: list[int] = [d.id for d in docs]
    texts: list[str] = [d.content for d in docs]

    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        vectors.extend([list(v) for v in embedder.embed(chunk)])

    if len(vectors) != len(ids):
        raise ValueError(f"Embedder returned {len(vectors)} vectors for {len(ids)} docs")

    vec_repo.rebuild(ids, vectors)
    return len(ids)


def delete_documents_multi_store(
    *,
    doc_repo: DocumentRepoPort,
    ids: Sequence[int],
    vec_repo: VectorRepoPort | None = None,
    embedder: EmbedderPort | None = None,
    embedder_factory: Callable[[], EmbedderPort] | None = None,
    rebuild_on_index_failure: bool = True,
) -> tuple[int, int | None, bool]:
    """
    Delete documents from SQL and, if provided, from the vector index.

    Returns: (deleted_sql, deleted_index, rebuilt_index)
    """
    ids_list = [int(x) for x in ids]
    if not ids_list:
        return 0, 0 if vec_repo else None, False

    resolved_embedder = embedder

    # Preflight vector mutability before SQL delete when rebuild fallback would require
    # an embedder we don't have yet. This avoids deleting SQL first and then discovering
    # we can't repair dense/hybrid drift.
    if vec_repo is not None and rebuild_on_index_failure and resolved_embedder is None:
        try:
            vec_repo.delete([])
        except Exception as preflight_err:
            if embedder_factory is not None:
                try:
                    resolved_embedder = embedder_factory()
                except Exception:
                    resolved_embedder = None
            if resolved_embedder is None:
                raise RuntimeError(
                    "Vector index preflight failed and no embeddings backend is available for "
                    "rebuild fallback. Aborting SQL delete to avoid multi-store drift."
                ) from preflight_err

    # SQL delete first. If this succeeds but index delete fails, we can rebuild index from DB.
    before = len(list(doc_repo.get(ids_list)))
    doc_repo.delete_documents(ids_list)
    deleted_sql = before

    if vec_repo is None:
        return deleted_sql, None, False

    try:
        deleted_index_raw = vec_repo.delete(ids_list)
        deleted_index = int(deleted_index_raw) if deleted_index_raw is not None else len(ids_list)
        return deleted_sql, deleted_index, False
    except Exception as delete_err:
        if not rebuild_on_index_failure:
            raise
        if resolved_embedder is None and embedder_factory is not None:
            try:
                resolved_embedder = embedder_factory()
            except Exception:
                resolved_embedder = None
        if resolved_embedder is None:
            raise RuntimeError(
                "Multi-store inconsistency risk: SQL delete succeeded but vector index delete "
                "failed and no embeddings backend is available for rebuild fallback. "
                "Configure embeddings and run `rag-rebuild-index` / POST /api/index/rebuild."
            ) from delete_err
        try:
            rebuilt = rebuild_index_from_db(
                doc_repo=doc_repo, vec_repo=vec_repo, embedder=resolved_embedder
            )
            return deleted_sql, None, rebuilt >= 0
        except Exception as rebuild_err:
            raise RuntimeError(
                "Multi-store inconsistency risk: SQL delete succeeded but vector index sync failed. "
                "Run `rag-rebuild-index` / POST /api/index/rebuild to repair."
            ) from rebuild_err


def delete_external_ids_multi_store(
    *,
    doc_repo: ExternalIdDeleteRepoPort,
    external_ids: Sequence[str],
    vec_repo: VectorRepoPort | None = None,
    embedder: EmbedderPort | None = None,
    embedder_factory: Callable[[], EmbedderPort] | None = None,
    rebuild_on_index_failure: bool = True,
) -> tuple[int, int | None, list[str], int, bool]:
    """
    Delete documents by external_id from SQL (with tombstones) and sync vector index.

    Returns:
        (deleted_sql, deleted_index, missing_external_ids, tombstoned, rebuilt_index)
    """
    ext_ids = [str(x).strip() for x in external_ids if str(x).strip()]
    if not ext_ids:
        return 0, 0 if vec_repo else None, [], 0, False

    resolved_embedder = embedder

    # Preflight vector mutability before SQL delete when rebuild fallback would require
    # an embedder we don't have yet. This avoids deleting SQL first and then discovering
    # we can't repair dense/hybrid drift.
    if vec_repo is not None and rebuild_on_index_failure and resolved_embedder is None:
        try:
            vec_repo.delete([])
        except Exception as preflight_err:
            if embedder_factory is not None:
                try:
                    resolved_embedder = embedder_factory()
                except Exception:
                    resolved_embedder = None
            if resolved_embedder is None:
                raise RuntimeError(
                    "Vector index preflight failed and no embeddings backend is available for "
                    "rebuild fallback. Aborting SQL delete to avoid multi-store drift."
                ) from preflight_err

    # SQL delete+tombstone first. If index delete fails afterward, rebuild from DB.
    deleted_sql, deleted_ids, missing_external_ids, tombstoned = doc_repo.delete_by_external_ids(
        ext_ids
    )

    if vec_repo is None:
        return deleted_sql, None, missing_external_ids, tombstoned, False

    try:
        deleted_index_raw = vec_repo.delete(deleted_ids)
        deleted_index = (
            int(deleted_index_raw) if deleted_index_raw is not None else len(deleted_ids)
        )
        return deleted_sql, deleted_index, missing_external_ids, tombstoned, False
    except Exception as delete_err:
        if not rebuild_on_index_failure:
            raise
        if resolved_embedder is None and embedder_factory is not None:
            try:
                resolved_embedder = embedder_factory()
            except Exception:
                resolved_embedder = None
        if resolved_embedder is None:
            raise RuntimeError(
                "Multi-store inconsistency risk: SQL delete succeeded but vector index delete "
                "failed and no embeddings backend is available for rebuild fallback. "
                "Configure embeddings and run `rag-rebuild-index` / POST /api/index/rebuild."
            ) from delete_err
        try:
            rebuilt = rebuild_index_from_db(
                doc_repo=doc_repo, vec_repo=vec_repo, embedder=resolved_embedder
            )
            return deleted_sql, None, missing_external_ids, tombstoned, rebuilt >= 0
        except Exception as rebuild_err:
            raise RuntimeError(
                "Multi-store inconsistency risk: SQL delete succeeded but vector index sync failed. "
                "Run `rag-rebuild-index` / POST /api/index/rebuild to repair."
            ) from rebuild_err

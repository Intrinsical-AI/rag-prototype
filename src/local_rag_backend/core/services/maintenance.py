"""
Maintenance operations that must keep SQL + vector index consistent.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Protocol

    from local_rag_backend.core.domain.types import DocId
    from local_rag_backend.core.ports import DocumentRepoPort, EmbedderPort, VectorRepoPort

    class ExternalIdDeleteRepoPort(DocumentRepoPort, Protocol):
        def delete_by_external_ids(
            self, external_ids: Sequence[str]
        ) -> tuple[int, list[DocId], list[str], int]: ...


@dataclass(frozen=True)
class MultiStoreDeleteResult:
    deleted_sql: int
    deleted_index: int | None
    rebuilt: bool


@dataclass(frozen=True)
class MultiStoreExternalIdDeleteResult:
    deleted_sql: int
    deleted_index: int | None
    missing_external_ids: list[str]
    tombstoned: int
    rebuilt: bool


_PREFLIGHT_MSG = (
    "Vector index preflight failed and no embeddings backend is available for "
    "rebuild fallback. Aborting SQL delete to avoid multi-store drift."
)
_CONSISTENCY_MSG = (
    "Multi-store inconsistency risk: SQL delete succeeded but vector index delete "
    "failed and no embeddings backend is available for rebuild fallback. "
    "Configure embeddings and run `rag-rebuild-index` / POST /api/index/rebuild."
)
_REBUILD_FAIL_MSG = (
    "Multi-store inconsistency risk: SQL delete succeeded but vector index sync failed. "
    "Run `rag-rebuild-index` / POST /api/index/rebuild to repair."
)


def _resolve_embedder_or_raise(
    *,
    current: EmbedderPort | None,
    factory: Callable[[], EmbedderPort] | None,
    cause: Exception,
    message: str,
) -> EmbedderPort:
    """Try *factory* when *current* is ``None``; raise ``RuntimeError(message)`` on failure."""
    if current is None and factory is not None:
        with contextlib.suppress(Exception):
            current = factory()
    if current is None:
        raise RuntimeError(message) from cause
    return current


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

    ids = [d.id for d in docs]
    texts = [d.content for d in docs]

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
    ids: Sequence[DocId],
    vec_repo: VectorRepoPort | None = None,
    embedder: EmbedderPort | None = None,
    embedder_factory: Callable[[], EmbedderPort] | None = None,
    rebuild_on_index_failure: bool = True,
) -> MultiStoreDeleteResult:
    """Delete documents from SQL and, if provided, from the vector index."""
    ids_list = [x for x in ids if str(x).strip()]
    if not ids_list:
        return MultiStoreDeleteResult(
            deleted_sql=0, deleted_index=0 if vec_repo else None, rebuilt=False
        )

    resolved_embedder = embedder

    if vec_repo is not None and rebuild_on_index_failure and resolved_embedder is None:
        try:
            _ = vec_repo.ntotal
        except Exception as preflight_err:
            resolved_embedder = _resolve_embedder_or_raise(
                current=resolved_embedder,
                factory=embedder_factory,
                cause=preflight_err,
                message=_PREFLIGHT_MSG,
            )

    before = len(list(doc_repo.get(ids_list)))
    doc_repo.delete_documents(ids_list)
    deleted_sql = before

    if vec_repo is None:
        return MultiStoreDeleteResult(deleted_sql=deleted_sql, deleted_index=None, rebuilt=False)

    try:
        deleted_index = vec_repo.delete(ids_list)
        return MultiStoreDeleteResult(
            deleted_sql=deleted_sql, deleted_index=deleted_index, rebuilt=False
        )
    except Exception as delete_err:
        if not rebuild_on_index_failure:
            raise
        resolved_embedder = _resolve_embedder_or_raise(
            current=resolved_embedder,
            factory=embedder_factory,
            cause=delete_err,
            message=_CONSISTENCY_MSG,
        )
        try:
            rebuilt = rebuild_index_from_db(
                doc_repo=doc_repo, vec_repo=vec_repo, embedder=resolved_embedder
            )
            return MultiStoreDeleteResult(
                deleted_sql=deleted_sql, deleted_index=None, rebuilt=rebuilt >= 0
            )
        except Exception as rebuild_err:
            raise RuntimeError(_REBUILD_FAIL_MSG) from rebuild_err


def delete_external_ids_multi_store(
    *,
    doc_repo: ExternalIdDeleteRepoPort,
    external_ids: Sequence[str],
    vec_repo: VectorRepoPort | None = None,
    embedder: EmbedderPort | None = None,
    embedder_factory: Callable[[], EmbedderPort] | None = None,
    rebuild_on_index_failure: bool = True,
) -> MultiStoreExternalIdDeleteResult:
    """Delete documents by external_id from SQL (with tombstones) and sync vector index."""
    ext_ids = [str(x).strip() for x in external_ids if str(x).strip()]
    if not ext_ids:
        return MultiStoreExternalIdDeleteResult(
            deleted_sql=0,
            deleted_index=0 if vec_repo else None,
            missing_external_ids=[],
            tombstoned=0,
            rebuilt=False,
        )

    resolved_embedder = embedder

    if vec_repo is not None and rebuild_on_index_failure and resolved_embedder is None:
        try:
            _ = vec_repo.ntotal
        except Exception as preflight_err:
            resolved_embedder = _resolve_embedder_or_raise(
                current=resolved_embedder,
                factory=embedder_factory,
                cause=preflight_err,
                message=_PREFLIGHT_MSG,
            )

    deleted_sql, deleted_ids, missing_external_ids, tombstoned = doc_repo.delete_by_external_ids(
        ext_ids
    )

    if vec_repo is None:
        return MultiStoreExternalIdDeleteResult(
            deleted_sql=deleted_sql,
            deleted_index=None,
            missing_external_ids=missing_external_ids,
            tombstoned=tombstoned,
            rebuilt=False,
        )

    try:
        deleted_index = vec_repo.delete(deleted_ids)
        return MultiStoreExternalIdDeleteResult(
            deleted_sql=deleted_sql,
            deleted_index=deleted_index,
            missing_external_ids=missing_external_ids,
            tombstoned=tombstoned,
            rebuilt=False,
        )
    except Exception as delete_err:
        if not rebuild_on_index_failure:
            raise
        resolved_embedder = _resolve_embedder_or_raise(
            current=resolved_embedder,
            factory=embedder_factory,
            cause=delete_err,
            message=_CONSISTENCY_MSG,
        )
        try:
            rebuilt = rebuild_index_from_db(
                doc_repo=doc_repo, vec_repo=vec_repo, embedder=resolved_embedder
            )
            return MultiStoreExternalIdDeleteResult(
                deleted_sql=deleted_sql,
                deleted_index=None,
                missing_external_ids=missing_external_ids,
                tombstoned=tombstoned,
                rebuilt=rebuilt >= 0,
            )
        except Exception as rebuild_err:
            raise RuntimeError(_REBUILD_FAIL_MSG) from rebuild_err

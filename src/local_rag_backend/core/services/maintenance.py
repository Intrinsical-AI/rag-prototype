"""
Maintenance operations that must keep SQL + vector index consistent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from local_rag_backend.core.ports import DocumentRepoPort, EmbedderPort, VectorRepoPort


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
    rebuild_on_index_failure: bool = True,
) -> tuple[int, int | None, bool]:
    """
    Delete documents from SQL and, if provided, from the vector index.

    Returns: (deleted_sql, deleted_index, rebuilt_index)
    """
    ids_list = [int(x) for x in ids]
    if not ids_list:
        return 0, 0 if vec_repo else None, False

    # SQL delete first. If this succeeds but index delete fails, we can rebuild index from DB.
    before = len(list(doc_repo.get(ids_list)))
    doc_repo.delete_documents(ids_list)
    deleted_sql = before

    if vec_repo is None:
        return deleted_sql, None, False

    try:
        vec_repo.delete(ids_list)
        return deleted_sql, len(ids_list), False
    except Exception:
        if not rebuild_on_index_failure:
            raise
        if embedder is None:
            raise
        rebuilt = rebuild_index_from_db(doc_repo=doc_repo, vec_repo=vec_repo, embedder=embedder)
        return deleted_sql, None, rebuilt >= 0

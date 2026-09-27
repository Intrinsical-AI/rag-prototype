"""Rebuild the vector index from canonical documents."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

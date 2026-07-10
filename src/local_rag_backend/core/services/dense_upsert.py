"""
Shared dense/hybrid upsert helpers.

These helpers centralize two critical operations used by API and CLI mutating flows:
- precomputing embeddings for all candidate upserts before SQL upsert
- syncing FAISS after SQL upsert with rebuild fallback
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from local_rag_backend.core.services.maintenance import rebuild_index_from_db

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from local_rag_backend.core.domain.types import DocId
    from local_rag_backend.core.ports import DocumentRepoPort, EmbedderPort, VectorRepoPort


class UpsertItemLike(Protocol):
    @property
    def external_id(self) -> str: ...

    @property
    def content(self) -> str: ...


class UpsertResultLike(Protocol):
    @property
    def external_id(self) -> str: ...

    @property
    def id(self) -> DocId: ...

    @property
    def content_changed(self) -> bool: ...


class ExistingStateLookupLike(Protocol):
    @property
    def content(self) -> str: ...

    @property
    def content_sha256(self) -> str | None: ...


class ExistingStateRepoLike(Protocol):
    def get_existing_doc_states_by_external_id(
        self, external_ids: Sequence[str]
    ) -> Mapping[str, ExistingStateLookupLike]: ...


def precompute_vectors_for_changed_items(
    *,
    items: Sequence[UpsertItemLike],
    doc_repo: ExistingStateRepoLike,
    embedder: EmbedderPort,
) -> dict[str, list[float]]:
    """
    Precompute vectors for every requested upsert.

    Safety takes precedence over avoiding redundant embedding calls here. Another mutation may
    update the same external_id between precompute and the later locked SQL/vector application,
    turning an apparently unchanged item into a real content update.
    """
    if not items:
        return {}
    _ = doc_repo
    to_embed = list(items)

    embedded = embedder.embed([it.content.strip() for it in to_embed])
    if len(embedded) != len(to_embed):
        raise RuntimeError(
            f"Embedder returned {len(embedded)} vectors for {len(to_embed)} documents."
        )
    return {it.external_id: list(vec) for it, vec in zip(to_embed, embedded, strict=False)}


def sync_dense_after_upsert(
    *,
    results: Sequence[UpsertResultLike],
    updated_content_ids: Sequence[DocId],
    vectors_by_external_id: Mapping[str, Sequence[float]],
    vec_repo: VectorRepoPort,
    doc_repo: DocumentRepoPort,
    embedder: EmbedderPort,
    rebuild_fn: Callable[..., int] | None = None,
) -> bool:
    """
    Sync vector store from SQL upsert result with rebuild fallback.

    Returns:
    - `False` when incremental sync succeeds or no content changed
    - `True` when fallback rebuild was required/succeeded
    """
    changed_results = [r for r in results if r.content_changed]
    if not changed_results:
        return False

    missing_vectors = [
        r.external_id for r in changed_results if r.external_id not in vectors_by_external_id
    ]
    if missing_vectors:
        raise RuntimeError(
            "Missing precomputed vectors for changed documents: " + ", ".join(missing_vectors[:10])
        )

    ids = [r.id for r in changed_results]
    vectors = [list(vectors_by_external_id[r.external_id]) for r in changed_results]
    try:
        if updated_content_ids:
            vec_repo.delete(updated_content_ids)
        vec_repo.upsert(ids, vectors)
        return False
    except Exception:
        rebuild = rebuild_fn or rebuild_index_from_db
        rebuilt = rebuild(doc_repo=doc_repo, vec_repo=vec_repo, embedder=embedder)
        return rebuilt >= 0

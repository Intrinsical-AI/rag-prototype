"""
Application-layer document/index mutation use cases.

This module centralizes business orchestration for `/api/docs*` flows so transport
handlers remain thin and focused on HTTP concerns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from local_rag_backend.app.services.results import (
    DeleteDocsByExternalIdSummary,
    DeleteDocsSummary,
    UpsertDocResult,
    UpsertDocsSummary,
)
from local_rag_backend.core.services.chunking import chunk_chars_v1
from local_rag_backend.core.services.dedup import chunk_dedup_sha256
from local_rag_backend.core.services.ingestion import (
    build_preprocess_fn_from_settings,
    default_formatter,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from local_rag_backend.app.services.ports import DocsMutationPorts
    from local_rag_backend.settings import Settings


class UpsertDocInputLike(Protocol):
    @property
    def external_id(self) -> str: ...

    @property
    def content(self) -> str: ...

    @property
    def source_id(self) -> str | None: ...

    @property
    def metadata(self) -> Mapping[str, Any] | None: ...


class TombstonedExternalIdsError(ValueError):
    """Raised when upsert attempts to reuse tombstoned external IDs."""

    def __init__(self, tombstoned: set[str]):
        self.tombstoned = set(tombstoned)
        super().__init__(
            f"Some external_id values are tombstoned (deleted): {sorted(self.tombstoned)[:10]}"
        )


def _embedding_model_name_for_dedup(settings_obj: Settings) -> str:
    if settings_obj.retrieval_mode not in ("dense", "hybrid"):
        return "none"
    if settings_obj.openai_api_key:
        return str(settings_obj.openai_embedding_model)
    return str(settings_obj.st_embedding_model)


def _uses_vector_index(settings_obj: Settings) -> bool:
    return settings_obj.retrieval_mode in ("dense", "hybrid")


def _build_vector_repo(
    *,
    settings_obj: Settings,
    ports: DocsMutationPorts,
    dim: int | None,
) -> Any:
    return ports.vector_repo_factory(
        index_path=settings_obj.index_path,
        id_map_path=settings_obj.id_map_path,
        dim=dim,
    )


def _precompute_vectors_if_needed(
    *,
    settings_obj: Settings,
    ports: DocsMutationPorts,
    doc_repo: Any,
    items: Sequence[Any],
) -> tuple[Any | None, dict[str, list[float]]]:
    if not _uses_vector_index(settings_obj):
        return None, {}

    embedder = ports.build_embedder()
    vectors_by_external_id = ports.precompute_vectors_fn(
        items=items,
        doc_repo=doc_repo,
        embedder=embedder,
    )
    return embedder, vectors_by_external_id


def ingest_docs_sync(
    *,
    texts: Sequence[str],
    settings_obj: Settings,
    ports: DocsMutationPorts,
) -> list[int]:
    if not texts:
        return []

    doc_repo = ports.doc_repo_factory()
    preprocess_fn = build_preprocess_fn_from_settings(settings_obj)
    chunker_version = str(settings_obj.ingest_chunker_version)
    embed_model = _embedding_model_name_for_dedup(settings_obj)

    source_id = f"api:/docs:v={chunker_version}:emb={embed_model}"
    unique_extids: list[str] = []
    items_by_extid: dict[str, Any] = {}

    for i, raw in enumerate(texts):
        md_base: dict[str, object] = {"source": "api:/docs", "input_index": i}
        processed = preprocess_fn(raw, md_base)
        chunks = chunk_chars_v1(
            processed,
            max_chars=settings_obj.ingest_chunk_chars,
            overlap=settings_obj.ingest_chunk_overlap,
        )
        for c in chunks:
            dedup = chunk_dedup_sha256(
                cleaned_text=c.text,
                chunker_version=chunker_version,
                embedding_model_name=embed_model,
            )
            external_id = f"chunk:{dedup}"

            if external_id in items_by_extid:
                continue
            unique_extids.append(external_id)

            md = dict(md_base)
            md["chunk_index"] = int(c.chunk_index)
            md["chunk_start_char"] = int(c.start_char)
            md["chunk_end_char"] = int(c.end_char)
            md["chunker_version"] = chunker_version
            md["embedding_model"] = embed_model
            md["dedup_sha256"] = dedup
            md["parent_doc_id"] = f"api:/docs:text={i}"

            content = default_formatter(c.text, md)
            items_by_extid[external_id] = ports.build_upsert_doc(
                external_id=external_id,
                content=content,
                source_id=source_id,
                metadata=md,
                chunk_dedup_sha256=dedup,
            )

    unique_items = list(items_by_extid.values())
    tombstoned = doc_repo.get_tombstoned_external_ids(unique_extids)
    if tombstoned:
        unique_extids = [e for e in unique_extids if e not in tombstoned]
        unique_items = [it for it in unique_items if it.external_id not in tombstoned]
    if not unique_items:
        return []

    embedder, vectors_by_external_id = _precompute_vectors_if_needed(
        settings_obj=settings_obj,
        ports=ports,
        doc_repo=doc_repo,
        items=unique_items,
    )

    results, _changed_content, updated_content_ids = doc_repo.upsert_documents_by_external_id(
        unique_items
    )
    id_by_ext = {r.external_id: int(r.id) for r in results}

    if embedder is not None:
        vec = _build_vector_repo(settings_obj=settings_obj, ports=ports, dim=embedder.dim)
        ports.sync_dense_fn(
            results=results,
            updated_content_ids=updated_content_ids,
            vectors_by_external_id=vectors_by_external_id,
            vec_repo=vec,
            doc_repo=doc_repo,
            embedder=embedder,
            rebuild_fn=ports.rebuild_fn,
        )

    return [id_by_ext[e] for e in unique_extids if e in id_by_ext]


def delete_docs_by_external_id_sync(
    *,
    external_ids: Sequence[str],
    settings_obj: Settings,
    ports: DocsMutationPorts,
) -> DeleteDocsByExternalIdSummary:
    ext_ids = [str(x).strip() for x in external_ids if str(x).strip()]
    if not ext_ids:
        return DeleteDocsByExternalIdSummary(
            deleted_sql=0,
            deleted_index=0,
            tombstoned=0,
            missing_external_ids=[],
            rebuilt_index=False,
        )

    doc_repo = ports.doc_repo_factory()
    if _uses_vector_index(settings_obj):
        vec = _build_vector_repo(settings_obj=settings_obj, ports=ports, dim=None)
        deleted_sql, deleted_index, missing, tombstoned, rebuilt = ports.delete_external_ids_fn(
            doc_repo=doc_repo,
            external_ids=ext_ids,
            vec_repo=vec,
            embedder_factory=ports.build_embedder,
            rebuild_on_index_failure=True,
        )
    else:
        deleted_sql, deleted_index, missing, tombstoned, rebuilt = ports.delete_external_ids_fn(
            doc_repo=doc_repo, external_ids=ext_ids
        )

    return DeleteDocsByExternalIdSummary(
        deleted_sql=deleted_sql,
        deleted_index=deleted_index,
        tombstoned=tombstoned,
        missing_external_ids=missing,
        rebuilt_index=rebuilt,
    )


def delete_docs_sync(
    *,
    ids: Sequence[int],
    settings_obj: Settings,
    ports: DocsMutationPorts,
) -> DeleteDocsSummary:
    ids_list = [int(i) for i in ids]
    if not ids_list:
        return DeleteDocsSummary(deleted_sql=0, deleted_index=0, rebuilt_index=False)

    doc_repo = ports.doc_repo_factory()
    if _uses_vector_index(settings_obj):
        vec = _build_vector_repo(settings_obj=settings_obj, ports=ports, dim=None)
        deleted_sql, deleted_index, rebuilt = ports.delete_docs_fn(
            doc_repo=doc_repo,
            vec_repo=vec,
            embedder_factory=ports.build_embedder,
            ids=ids_list,
            rebuild_on_index_failure=True,
        )
        return DeleteDocsSummary(
            deleted_sql=deleted_sql, deleted_index=deleted_index, rebuilt_index=rebuilt
        )

    deleted_sql, _, _ = ports.delete_docs_fn(doc_repo=doc_repo, ids=ids_list)
    return DeleteDocsSummary(deleted_sql=deleted_sql, deleted_index=None, rebuilt_index=False)


def upsert_docs_sync(
    *,
    docs: Sequence[UpsertDocInputLike],
    settings_obj: Settings,
    ports: DocsMutationPorts,
) -> UpsertDocsSummary:
    ext_ids = [d.external_id for d in docs]
    if len(set(ext_ids)) != len(ext_ids):
        raise ValueError("external_id values must be unique per request.")

    doc_repo = ports.doc_repo_factory()
    tombstoned = doc_repo.get_tombstoned_external_ids(ext_ids)
    if tombstoned:
        raise TombstonedExternalIdsError(set(tombstoned))

    items = [
        ports.build_upsert_doc(
            external_id=d.external_id,
            content=d.content,
            source_id=d.source_id,
            metadata=d.metadata,
        )
        for d in docs
    ]

    embedder, vectors_by_external_id = _precompute_vectors_if_needed(
        settings_obj=settings_obj,
        ports=ports,
        doc_repo=doc_repo,
        items=items,
    )

    results, _changed_content, updated_content_ids = doc_repo.upsert_documents_by_external_id(items)
    inserted = sum(1 for r in results if r.action == "inserted")
    updated = sum(1 for r in results if r.action == "updated")
    unchanged = sum(1 for r in results if r.action == "unchanged")

    rebuilt_index = False
    if embedder is not None:
        vec = _build_vector_repo(settings_obj=settings_obj, ports=ports, dim=embedder.dim)
        rebuilt_index = ports.sync_dense_fn(
            results=results,
            updated_content_ids=updated_content_ids,
            vectors_by_external_id=vectors_by_external_id,
            vec_repo=vec,
            doc_repo=doc_repo,
            embedder=embedder,
            rebuild_fn=ports.rebuild_fn,
        )

    return UpsertDocsSummary(
        inserted=inserted,
        updated=updated,
        unchanged=unchanged,
        rebuilt_index=rebuilt_index,
        results=[
            UpsertDocResult(
                external_id=r.external_id,
                id=int(r.id),
                action=r.action,
                content_changed=bool(r.content_changed),
            )
            for r in results
        ],
    )

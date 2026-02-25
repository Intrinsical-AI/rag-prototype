"""Application use case for ingesting raw texts into durable docs mutations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_backend.app.application.docs_mutation import (
    MutationCoordinator,
    MutationIntent,
    MutationUpsertInput,
)
from local_rag_backend.core.services.chunking import chunk_chars_v1
from local_rag_backend.core.services.dedup import chunk_dedup_sha256
from local_rag_backend.core.services.ingestion import (
    build_preprocess_fn_from_settings,
    default_formatter,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from local_rag_backend.app.contracts.ports import DocsMutationPorts
    from local_rag_backend.settings import Settings


def _embedding_model_name_for_dedup(settings_obj: Settings) -> str:
    if settings_obj.retrieval_mode not in ("dense", "hybrid"):
        return "none"
    if settings_obj.openai_api_key:
        return str(settings_obj.openai_embedding_model)
    return str(settings_obj.st_embedding_model)


def ingest_docs_sync(
    *,
    texts: Sequence[str],
    settings_obj: Settings,
    ports: DocsMutationPorts,
    source: str = "api:/docs",
) -> list[str]:
    if not texts:
        return []

    doc_repo = ports.doc_repo_factory()
    preprocess_fn = build_preprocess_fn_from_settings(settings_obj)
    chunker_version = str(settings_obj.ingest_chunker_version)
    embed_model = _embedding_model_name_for_dedup(settings_obj)

    source_id = f"{source}:v={chunker_version}:emb={embed_model}"
    unique_extids: list[str] = []
    items_by_extid: dict[str, MutationUpsertInput] = {}

    for i, raw in enumerate(texts):
        md_base: dict[str, object] = {"source": source, "input_index": i}
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
            md["parent_doc_id"] = f"{source}:text={i}"

            content = default_formatter(c.text, md)
            items_by_extid[external_id] = MutationUpsertInput(
                external_id=external_id,
                content=content,
                source_id=source_id,
                metadata=md,
            )

    unique_items = [items_by_extid[eid] for eid in unique_extids]
    tombstoned = doc_repo.get_tombstoned_external_ids(unique_extids)
    if tombstoned:
        unique_extids = [e for e in unique_extids if e not in tombstoned]
        unique_items = [it for it in unique_items if it.external_id not in tombstoned]
    if not unique_items:
        return []

    summary = MutationCoordinator(settings_obj=settings_obj, ports=ports).execute(
        MutationIntent(op_id="", upserts=tuple(unique_items), source=source)
    )
    id_by_ext = {r.external_id: r.id for r in list(summary.results or [])}
    return [id_by_ext[e] for e in unique_extids if e in id_by_ext]


__all__ = ["ingest_docs_sync"]

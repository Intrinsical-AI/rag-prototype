"""Internal handlers for docs mutation coordinator.

These functions keep SQL/vector/recovery concerns isolated so the coordinator
focuses on orchestration and journal state transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from local_rag_backend.core.domain.types import DocId
from local_rag_backend.core.ports.contracts import MutationRecord
from local_rag_backend.core.use_cases.results import UpsertDocResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from local_rag_backend.core.ports.contracts import (
        MutationJournalPort,
        MutationState,
        UpsertDocBuilderPort,
    )
    from local_rag_backend.settings import Settings


@dataclass(frozen=True)
class SqlMutationOutcome:
    inserted: int
    updated: int
    unchanged: int
    results: list[UpsertDocResult]
    changed_content: list[tuple[DocId, str]]
    updated_content_ids: list[DocId]
    deleted_sql: int
    deleted_doc_ids: list[DocId]
    tombstoned: int
    missing_external_ids: list[str]


def _normalize_str_items(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def capture_before_image(*, doc_repo: Any, intent: Any) -> dict[str, Any]:
    snapshots: list[dict[str, Any]] = []
    if intent.upserts or intent.delete_external_ids:
        external_ids = sorted(
            {
                *(u.external_id for u in intent.upserts),
                *intent.delete_external_ids,
            }
        )
        if hasattr(doc_repo, "snapshot_by_external_ids"):
            ext_snaps = list(doc_repo.snapshot_by_external_ids(external_ids))
            snapshots.extend([dict(cast("dict[str, Any]", s)) for s in ext_snaps])
    if intent.delete_ids and hasattr(doc_repo, "snapshot_by_ids"):
        id_snaps = list(doc_repo.snapshot_by_ids([DocId(str(x)) for x in intent.delete_ids]))
        snapshots.extend([dict(cast("dict[str, Any]", s)) for s in id_snaps])

    existing_tombstones: list[str] = []
    if intent.delete_external_ids and hasattr(doc_repo, "get_tombstoned_external_ids"):
        existing_tombstones = sorted(
            doc_repo.get_tombstoned_external_ids(list(intent.delete_external_ids))
        )

    # Deduplicate snapshots by doc_id (later captures overwrite earlier duplicates).
    by_doc_id = {str(s.get("id")): s for s in snapshots if str(s.get("id") or "").strip()}
    return {
        "docs": list(by_doc_id.values()),
        "existing_tombstones": existing_tombstones,
    }


def apply_sql_mutation(
    *,
    doc_repo: Any,
    intent: Any,
    build_upsert_doc: UpsertDocBuilderPort,
) -> SqlMutationOutcome:
    inserted = 0
    updated = 0
    unchanged = 0
    results: list[UpsertDocResult] = []
    changed_content: list[tuple[DocId, str]] = []
    updated_content_ids: list[DocId] = []
    deleted_sql = 0
    deleted_doc_ids: list[DocId] = []
    tombstoned = 0
    missing_external_ids: list[str] = []

    if intent.upserts:
        items = [
            build_upsert_doc(
                external_id=u.external_id,
                content=u.content,
                source_id=u.source_id,
                metadata=u.metadata,
            )
            for u in intent.upserts
        ]
        upsert_results, changed, updated_ids = doc_repo.upsert_documents_by_external_id(items)
        changed_content.extend([(DocId(str(doc_id)), str(content)) for doc_id, content in changed])
        updated_content_ids.extend([DocId(str(doc_id)) for doc_id in updated_ids])
        for r in upsert_results:
            action = str(r.action)
            if action == "inserted":
                inserted += 1
            elif action == "updated":
                updated += 1
            else:
                unchanged += 1
            results.append(
                UpsertDocResult(
                    external_id=str(r.external_id),
                    id=str(r.id),
                    action=action,
                    content_changed=bool(r.content_changed),
                )
            )

    if intent.delete_ids:
        ids_doc = [DocId(str(x)) for x in intent.delete_ids]
        existing_docs = list(doc_repo.get(ids_doc))
        deleted_sql += len(existing_docs)
        deleted_doc_ids.extend([DocId(str(d.id)) for d in existing_docs])
        if existing_docs:
            doc_repo.delete_documents(ids_doc)

    if intent.delete_external_ids:
        deleted_count, deleted_ids, missing, tombstoned_count = doc_repo.delete_by_external_ids(
            list(intent.delete_external_ids)
        )
        deleted_sql += int(deleted_count)
        deleted_doc_ids.extend([DocId(str(x)) for x in list(deleted_ids)])
        missing_external_ids.extend([str(x) for x in list(missing)])
        tombstoned += int(tombstoned_count)

    dedup_deleted_ids = list({str(x): DocId(str(x)) for x in deleted_doc_ids}.values())
    return SqlMutationOutcome(
        inserted=inserted,
        updated=updated,
        unchanged=unchanged,
        results=results,
        changed_content=changed_content,
        updated_content_ids=updated_content_ids,
        deleted_sql=deleted_sql,
        deleted_doc_ids=dedup_deleted_ids,
        tombstoned=tombstoned,
        missing_external_ids=missing_external_ids,
    )


def apply_vector_delta(
    *,
    sql_outcome: SqlMutationOutcome,
    doc_repo: Any,
    settings_obj: Settings,
    ports: Any,
    uses_vector_index: bool,
) -> tuple[int | None, int | None]:
    if not uses_vector_index:
        return None, None

    delete_ids = [
        DocId(str(x))
        for x in {
            *[str(doc_id) for doc_id in sql_outcome.updated_content_ids],
            *[str(doc_id) for doc_id in sql_outcome.deleted_doc_ids],
        }
    ]

    needs_upsert_vectors = bool(sql_outcome.changed_content)
    embedder = ports.build_embedder() if needs_upsert_vectors else None
    dim = int(embedder.dim) if embedder is not None else None
    vec_repo = ports.vector_repo_factory(
        index_path=settings_obj.index_path,
        id_map_path=settings_obj.id_map_path,
        dim=dim,
        backend=getattr(settings_obj, "vector_backend", "auto"),
    )

    upsert_vectors: list[tuple[DocId, list[float]]] = []
    if embedder is not None:
        embed_inputs = [content for _, content in sql_outcome.changed_content]
        vectors = embedder.embed(embed_inputs)
        if len(vectors) != len(embed_inputs):
            raise RuntimeError(
                f"Embedder returned {len(vectors)} vectors for {len(embed_inputs)} changed documents."
            )
        upsert_vectors = [
            (doc_id, list(vec))
            for (doc_id, _content), vec in zip(sql_outcome.changed_content, vectors, strict=False)
        ]

    apply_delta = getattr(vec_repo, "apply_delta_atomic", None)
    if not callable(apply_delta):
        raise RuntimeError(
            "Vector adapter must implement apply_delta_atomic for DURABLE_SAGA mutations."
        )
    apply_delta(delete_ids=delete_ids, upserts=upsert_vectors)

    index_doc_count = (
        len(list(doc_repo.get_all_documents())) if hasattr(doc_repo, "get_all_documents") else None
    )
    return len(delete_ids), index_doc_count


def rollback_sql(*, doc_repo: Any, intent: dict[str, Any], before_image: dict[str, Any]) -> None:
    upserts = [str(x.get("external_id") or "").strip() for x in list(intent.get("upserts") or [])]
    delete_external_ids = _normalize_str_items(list(intent.get("delete_external_ids") or []))
    affected_external_ids = _normalize_str_items([*upserts, *delete_external_ids])

    if affected_external_ids and hasattr(doc_repo, "hard_delete_by_external_ids"):
        doc_repo.hard_delete_by_external_ids(affected_external_ids)

    snapshots = list(before_image.get("docs") or [])
    if snapshots and hasattr(doc_repo, "restore_from_snapshots"):
        doc_repo.restore_from_snapshots(snapshots)

    if delete_external_ids and hasattr(doc_repo, "delete_tombstones"):
        existing_before = set(before_image.get("existing_tombstones") or [])
        created_now = [ext for ext in delete_external_ids if ext not in existing_before]
        if created_now:
            doc_repo.delete_tombstones(created_now)


def recover_record(
    *,
    journal: MutationJournalPort,
    record: MutationRecord,
    rollback_sql_fn: Any,
    doc_repo_factory: Any,
) -> None:
    if record.state in {"COMMITTED", "ROLLED_BACK"}:
        return
    if record.state == "VECTOR_COMMITTED":
        committed = _clone_record(record=record, state="COMMITTED")
        journal.upsert(committed)
        journal.delete(record.op_id)
        return

    doc_repo = doc_repo_factory()
    if record.before_image is None:
        rolled_back = _clone_record(
            record=record,
            state="ROLLED_BACK",
            before_image=None,
            error=record.error or "No before_image was available; record marked as rolled back.",
        )
        journal.upsert(rolled_back)
        return

    compensating = _clone_record(record=record, state="COMPENSATING")
    journal.upsert(compensating)
    try:
        rollback_sql_fn(doc_repo=doc_repo, intent=record.intent, before_image=record.before_image)
        rolled_back = _clone_record(record=record, state="ROLLED_BACK")
        journal.upsert(rolled_back)
    except Exception as rollback_err:
        failed = _clone_record(
            record=record, state="FAILED_NEEDS_RECOVERY", error=str(rollback_err)
        )
        journal.upsert(failed)


def _clone_record(
    *,
    record: MutationRecord,
    state: MutationState,
    before_image: dict[str, Any] | None | object = ...,
    error: str | None | object = ...,
) -> MutationRecord:
    return MutationRecord(
        op_id=record.op_id,
        state=state,
        intent=dict(record.intent),
        before_image=record.before_image
        if before_image is ...
        else cast("dict[str, Any] | None", before_image),
        outcome=record.outcome,
        error=record.error if error is ... else cast("str | None", error),
        attempts=int(record.attempts) + 1,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


__all__ = [
    "SqlMutationOutcome",
    "apply_sql_mutation",
    "apply_vector_delta",
    "capture_before_image",
    "recover_record",
    "rollback_sql",
]

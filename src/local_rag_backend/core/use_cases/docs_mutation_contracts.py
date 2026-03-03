"""Mutation intent and journal payload helpers.

This module contains transport-neutral shaping rules used by MutationCoordinator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from local_rag_backend.core.ports.contracts import MutationRecord
from local_rag_backend.core.use_cases.results import MutationSummary, UpsertDocResult

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass(frozen=True)
class MutationUpsertInput:
    external_id: str
    content: str
    source_id: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class MutationIntent:
    op_id: str
    upserts: tuple[MutationUpsertInput, ...] = ()
    delete_ids: tuple[str, ...] = ()
    delete_external_ids: tuple[str, ...] = ()
    source: str = "unknown"


def normalize_intent(
    *,
    intent: MutationIntent,
    new_op_id: Callable[[], str],
) -> MutationIntent:
    upserts = [
        MutationUpsertInput(
            external_id=str(it.external_id).strip(),
            content=str(it.content).strip(),
            source_id=(str(it.source_id) if it.source_id is not None else None),
            metadata=dict(it.metadata) if it.metadata is not None else None,
        )
        for it in list(intent.upserts)
        if str(it.external_id).strip() and str(it.content).strip()
    ]
    ext_ids = normalize_str_items(intent.delete_external_ids)
    delete_ids = normalize_str_items(intent.delete_ids)

    if not upserts and not ext_ids and not delete_ids:
        raise ValueError("Mutation intent must include upserts and/or deletions.")

    upsert_ext_ids = [it.external_id for it in upserts]
    if len(set(upsert_ext_ids)) != len(upsert_ext_ids):
        raise ValueError("external_id values in upserts must be unique per mutation intent.")
    ext_conflict = sorted(set(upsert_ext_ids) & set(ext_ids))
    if ext_conflict:
        raise ValueError(
            "upserts and delete_external_ids cannot target the same external_id values: "
            + ", ".join(ext_conflict[:10])
        )

    return MutationIntent(
        op_id=str(intent.op_id).strip() or new_op_id(),
        upserts=tuple(upserts),
        delete_ids=tuple(delete_ids),
        delete_external_ids=tuple(ext_ids),
        source=str(intent.source or "unknown"),
    )


def summary_from_record(record: MutationRecord) -> MutationSummary:
    payload = dict(record.outcome or {})
    return MutationSummary(
        op_id=str(payload.get("op_id") or record.op_id),
        inserted=int(payload.get("inserted") or 0),
        updated=int(payload.get("updated") or 0),
        unchanged=int(payload.get("unchanged") or 0),
        deleted_sql=int(payload.get("deleted_sql") or 0),
        deleted_index=(
            int(payload["deleted_index"]) if payload.get("deleted_index") is not None else None
        ),
        tombstoned=int(payload.get("tombstoned") or 0),
        missing_external_ids=list(payload.get("missing_external_ids") or []),
        index_rebuilt=bool(payload.get("index_rebuilt") or False),
        index_doc_count=(
            int(payload["index_doc_count"]) if payload.get("index_doc_count") is not None else None
        ),
        results=[
            UpsertDocResult(
                external_id=str(item.get("external_id") or ""),
                id=str(item.get("id") or ""),
                action=str(item.get("action") or "unchanged"),
                content_changed=bool(item.get("content_changed") or False),
            )
            for item in list(payload.get("results") or [])
            if isinstance(item, dict)
        ],
    )


def summary_to_payload(summary: MutationSummary) -> dict[str, Any]:
    return {
        "op_id": summary.op_id,
        "inserted": summary.inserted,
        "updated": summary.updated,
        "unchanged": summary.unchanged,
        "deleted_sql": summary.deleted_sql,
        "deleted_index": summary.deleted_index,
        "tombstoned": summary.tombstoned,
        "missing_external_ids": list(summary.missing_external_ids or []),
        "index_rebuilt": bool(summary.index_rebuilt),
        "index_doc_count": summary.index_doc_count,
        "results": [
            {
                "external_id": r.external_id,
                "id": r.id,
                "action": r.action,
                "content_changed": bool(r.content_changed),
            }
            for r in list(summary.results or [])
        ],
    }


def intent_to_dict(intent: MutationIntent) -> dict[str, Any]:
    return {
        "op_id": intent.op_id,
        "source": intent.source,
        "upserts": [
            {
                "external_id": u.external_id,
                "content": u.content,
                "source_id": u.source_id,
                "metadata": dict(u.metadata) if u.metadata is not None else None,
            }
            for u in intent.upserts
        ],
        "delete_ids": list(intent.delete_ids),
        "delete_external_ids": list(intent.delete_external_ids),
    }


def normalize_str_items(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def canonical_intent_payload(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return {
            str(k): canonical_intent_payload(v)
            for k, v in sorted(payload.items(), key=lambda item: str(item[0]))
        }
    if isinstance(payload, (list, tuple)):
        return [canonical_intent_payload(v) for v in payload]
    if isinstance(payload, set):
        return sorted(canonical_intent_payload(v) for v in payload)
    return payload


__all__ = [
    "MutationIntent",
    "MutationUpsertInput",
    "canonical_intent_payload",
    "intent_to_dict",
    "normalize_intent",
    "normalize_str_items",
    "summary_from_record",
    "summary_to_payload",
]

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
    scope: str | None = None
    snapshot_id: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class MutationIntent:
    op_id: str
    upserts: tuple[MutationUpsertInput, ...] = ()
    delete_ids: tuple[str, ...] = ()
    delete_external_ids: tuple[str, ...] = ()
    hard_delete_external_ids: tuple[str, ...] = ()
    source: str = "unknown"


def normalize_intent(
    *,
    intent: MutationIntent,
    new_op_id: Callable[[], str],
) -> MutationIntent:
    upserts: list[MutationUpsertInput] = []
    for idx, item in enumerate(intent.upserts):
        external_id = str(item.external_id).strip()
        content = str(item.content)
        if not external_id:
            raise ValueError(f"upserts[{idx}].external_id must not be blank")
        if not content.strip():
            raise ValueError(f"upserts[{idx}].content must not be blank")
        upserts.append(
            MutationUpsertInput(
                external_id=external_id,
                content=content,
                source_id=(str(item.source_id) if item.source_id is not None else None),
                scope=(str(item.scope) if item.scope is not None else None),
                snapshot_id=(str(item.snapshot_id) if item.snapshot_id is not None else None),
                metadata=dict(item.metadata) if item.metadata is not None else None,
            )
        )
    ext_ids = normalize_str_items(intent.delete_external_ids)
    hard_ext_ids = normalize_str_items(intent.hard_delete_external_ids)
    delete_ids = normalize_str_items(intent.delete_ids)

    if not upserts and not ext_ids and not hard_ext_ids and not delete_ids:
        raise ValueError("Mutation intent must include upserts and/or deletions.")

    upsert_ext_ids = [it.external_id for it in upserts]
    if len(set(upsert_ext_ids)) != len(upsert_ext_ids):
        raise ValueError("external_id values in upserts must be unique per mutation intent.")
    ext_conflict = sorted(
        (set(upsert_ext_ids) & set(ext_ids))
        | (set(upsert_ext_ids) & set(hard_ext_ids))
        | (set(ext_ids) & set(hard_ext_ids))
    )
    if ext_conflict:
        raise ValueError(
            "upserts and deletion modes cannot target the same external_id values: "
            + ", ".join(ext_conflict[:10])
        )

    return MutationIntent(
        op_id=str(intent.op_id).strip() or new_op_id(),
        upserts=tuple(upserts),
        delete_ids=tuple(delete_ids),
        delete_external_ids=tuple(ext_ids),
        hard_delete_external_ids=tuple(hard_ext_ids),
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
    payload = {
        "op_id": intent.op_id,
        "source": intent.source,
        "upserts": [
            {
                "external_id": u.external_id,
                "content": u.content,
                "source_id": u.source_id,
                "scope": u.scope,
                "snapshot_id": u.snapshot_id,
                "metadata": dict(u.metadata) if u.metadata is not None else None,
            }
            for u in intent.upserts
        ],
        "delete_ids": list(intent.delete_ids),
        "delete_external_ids": list(intent.delete_external_ids),
    }
    if intent.hard_delete_external_ids:
        payload["hard_delete_external_ids"] = list(intent.hard_delete_external_ids)
    return payload


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

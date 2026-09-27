"""Filesystem-backed mutation journal used for durable multi-store sagas."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, cast

from local_rag_backend.core.errors import MutationRecoveryRequiredError
from local_rag_backend.core.ports.contracts import (
    MutationJournalPort,
    MutationRecord,
    MutationState,
)
from local_rag_backend.infrastructure.persistence.shared.atomic_io import atomic_write_text

_JOURNAL_VERSION = 2
_MUTATION_STATES = {
    "PREPARED",
    "SQL_COMMITTED",
    "VECTOR_COMMITTED",
    "COMMITTED",
    "COMPENSATING",
    "ROLLED_BACK",
    "FAILED_NEEDS_RECOVERY",
}


def _record_path(root: Path, op_id: str) -> Path:
    raw = str(op_id).strip()
    if not raw:
        raise ValueError("op_id must not be blank")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return root / f"{digest}.json"


def _record_from_dict(obj: dict[str, Any]) -> MutationRecord:
    if obj.get("schema_version") != _JOURNAL_VERSION:
        raise ValueError("Unsupported mutation journal schema_version; expected 2.")
    state = obj.get("state")
    if not isinstance(state, str) or state not in _MUTATION_STATES:
        raise ValueError(f"Invalid mutation journal state {state!r}.")
    op_id = obj.get("op_id")
    if not isinstance(op_id, str) or not op_id.strip():
        raise ValueError("Mutation journal record has empty or invalid op_id.")
    if not isinstance(obj.get("intent"), dict):
        raise ValueError("Mutation journal intent must be an object.")
    for field in ("before_image", "outcome"):
        if obj.get(field) is not None and not isinstance(obj[field], dict):
            raise ValueError(f"Mutation journal {field} must be an object or null.")
    if type(obj.get("vector_attempted")) is not bool:
        raise ValueError("Mutation journal vector_attempted must be a boolean.")
    record = MutationRecord(
        op_id=op_id,
        state=cast("MutationState", state),
        intent=dict(obj["intent"]),
        before_image=obj.get("before_image"),
        outcome=obj.get("outcome"),
        error=(str(obj["error"]) if obj.get("error") is not None else None),
        attempts=int(obj.get("attempts") or 0),
        created_at=float(obj.get("created_at") or 0.0),
        updated_at=float(obj.get("updated_at") or 0.0),
        vector_attempted=obj["vector_attempted"],
    )
    if state in {"COMMITTED", "VECTOR_COMMITTED"}:
        record.validate_outcome()
    else:
        record.validate_before_image()
    return record


def _record_to_dict(record: MutationRecord) -> dict[str, Any]:
    return {
        "schema_version": _JOURNAL_VERSION,
        "op_id": record.op_id,
        "state": record.state,
        "intent": dict(record.intent),
        "before_image": dict(record.before_image) if record.before_image is not None else None,
        "outcome": dict(record.outcome) if record.outcome is not None else None,
        "error": record.error,
        "attempts": int(record.attempts),
        "created_at": float(record.created_at),
        "updated_at": float(record.updated_at),
        "vector_attempted": record.vector_attempted,
    }


def _read_record(path: Path) -> MutationRecord:
    data: Any = None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Record payload is not a JSON object.")
        record = _record_from_dict(data)
        if _record_path(path.parent, record.op_id) != path:
            raise ValueError("Mutation journal filename does not match op_id.")
        return record
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        identity = (
            f" op_id={data.get('op_id')!r} state={data.get('state')!r}"
            if isinstance(data, dict)
            else ""
        )
        raise MutationRecoveryRequiredError(
            f"Unreadable mutation journal record at {path}{identity}: {exc} "
            "The record was preserved; an index rebuild cannot repair the journal."
        ) from exc


class FileMutationJournal(MutationJournalPort):
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, op_id: str) -> MutationRecord | None:
        path = _record_path(self.root, op_id)
        return _read_record(path) if path.is_file() else None

    def upsert(self, record: MutationRecord) -> None:
        now = time.time()
        existing = self.get(record.op_id)
        normalized = MutationRecord(
            op_id=record.op_id,
            state=record.state,
            intent=dict(record.intent),
            before_image=(dict(record.before_image) if record.before_image is not None else None),
            outcome=(dict(record.outcome) if record.outcome is not None else None),
            error=record.error,
            attempts=max(
                int(record.attempts), int(existing.attempts) if existing is not None else 0
            ),
            created_at=float(
                record.created_at or (existing.created_at if existing is not None else now)
            ),
            updated_at=now,
            vector_attempted=record.vector_attempted,
        )
        path = _record_path(self.root, normalized.op_id)
        try:
            atomic_write_text(
                path,
                json.dumps(_record_to_dict(normalized), ensure_ascii=True, separators=(",", ":")),
            )
        except OSError as exc:
            raise MutationRecoveryRequiredError(
                f"Cannot persist mutation journal at {path} "
                f"op_id={record.op_id!r} state={record.state!r}: {exc}"
            ) from exc

    def list_incomplete(self, *, limit: int = 100) -> list[MutationRecord]:
        if limit <= 0:
            raise ValueError("Mutation recovery limit must be positive.")
        out: list[MutationRecord] = []
        for path in sorted(self.root.glob("*.json")):
            record = _read_record(path)
            if record.state not in {"COMMITTED", "ROLLED_BACK"}:
                out.append(record)
                if len(out) >= limit:
                    break
        return out


__all__ = ["FileMutationJournal"]

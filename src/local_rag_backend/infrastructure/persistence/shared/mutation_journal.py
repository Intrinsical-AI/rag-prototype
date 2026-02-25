"""Filesystem-backed mutation journal used for durable multi-store sagas."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

from local_rag_backend.app.contracts.ports import (
    MutationJournalPort,
    MutationRecord,
    MutationState,
)
from local_rag_backend.infrastructure.persistence.shared.atomic_io import atomic_write_text

logger = logging.getLogger(__name__)
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
    state_raw = str(obj.get("state") or "PREPARED")
    if state_raw not in _MUTATION_STATES:
        state_raw = "PREPARED"
    return MutationRecord(
        op_id=str(obj.get("op_id") or ""),
        state=cast("MutationState", state_raw),
        intent=dict(obj.get("intent") or {}),
        before_image=(
            dict(obj["before_image"]) if isinstance(obj.get("before_image"), dict) else None
        ),
        outcome=dict(obj["outcome"]) if isinstance(obj.get("outcome"), dict) else None,
        error=(str(obj["error"]) if obj.get("error") is not None else None),
        attempts=int(obj.get("attempts") or 0),
        created_at=float(obj.get("created_at") or 0.0),
        updated_at=float(obj.get("updated_at") or 0.0),
    )


def _record_to_dict(record: MutationRecord) -> dict[str, Any]:
    return {
        "op_id": record.op_id,
        "state": record.state,
        "intent": dict(record.intent),
        "before_image": dict(record.before_image) if record.before_image is not None else None,
        "outcome": dict(record.outcome) if record.outcome is not None else None,
        "error": record.error,
        "attempts": int(record.attempts),
        "created_at": float(record.created_at),
        "updated_at": float(record.updated_at),
    }


class FileMutationJournal(MutationJournalPort):
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, op_id: str) -> MutationRecord | None:
        path = _record_path(self.root, op_id)
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid mutation journal record at {path}")
        record = _record_from_dict(data)
        if not record.op_id:
            raise ValueError(f"Mutation record at {path} has empty op_id")
        return record

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
            updated_at=float(now),
        )
        path = _record_path(self.root, normalized.op_id)
        atomic_write_text(
            path,
            json.dumps(_record_to_dict(normalized), ensure_ascii=True, separators=(",", ":")),
        )

    def delete(self, op_id: str) -> None:
        path = _record_path(self.root, op_id)
        with suppress(FileNotFoundError):
            path.unlink()

    def list_incomplete(self, *, limit: int = 100) -> list[MutationRecord]:
        out: list[MutationRecord] = []
        for path in sorted(self.root.glob("*.json")):
            if len(out) >= int(limit):
                break
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning(
                    "Skipping unreadable mutation journal record at %s", path, exc_info=True
                )
                continue
            if not isinstance(data, dict):
                continue
            rec = _record_from_dict(data)
            if rec.state in {"COMMITTED", "ROLLED_BACK"}:
                continue
            out.append(rec)
        return out


__all__ = ["FileMutationJournal"]

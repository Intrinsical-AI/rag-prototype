"""Filesystem-backed mutation journal used for durable multi-store sagas."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from local_rag_backend.core.errors import MutationRecoveryRequiredError
from local_rag_backend.core.ports.contracts import (
    MutationJournalPort,
    MutationRecord,
    MutationState,
)
from local_rag_backend.infrastructure.persistence.shared.atomic_io import (
    atomic_write_text,
    ensure_durable_directory,
    fsync_directory,
)

_JOURNAL_VERSION = 2
_DONE_MAX_AGE_S = 30 * 24 * 60 * 60
_DONE_MAX_BYTES = 256 * 1024 * 1024
_DONE_SWEEP_INTERVAL_S = 60 * 60
_TERMINAL_STATES = {"COMMITTED", "ROLLED_BACK"}
_UNVERSIONED_TERMINAL_KEYS = {
    "op_id",
    "state",
    "intent",
    "before_image",
    "outcome",
    "error",
    "attempts",
    "created_at",
    "updated_at",
}
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


def _record_from_dict(
    obj: dict[str, Any], *, allow_unversioned_terminal: bool = False
) -> MutationRecord:
    state = obj.get("state")
    if not isinstance(state, str) or state not in _MUTATION_STATES:
        raise ValueError(f"Invalid mutation journal state {state!r}.")
    unversioned = "schema_version" not in obj
    if unversioned:
        if (
            not allow_unversioned_terminal
            or state not in _TERMINAL_STATES
            or set(obj) != _UNVERSIONED_TERMINAL_KEYS
        ):
            raise ValueError("Unsupported unversioned mutation journal record.")
    elif obj["schema_version"] != _JOURNAL_VERSION:
        raise ValueError("Unsupported mutation journal schema_version; expected 2.")
    op_id = obj.get("op_id")
    if not isinstance(op_id, str) or not op_id.strip():
        raise ValueError("Mutation journal record has empty or invalid op_id.")
    if not isinstance(obj.get("intent"), dict):
        raise ValueError("Mutation journal intent must be an object.")
    for field in ("before_image", "outcome"):
        if obj.get(field) is not None and not isinstance(obj[field], dict):
            raise ValueError(f"Mutation journal {field} must be an object or null.")
    if not unversioned and type(obj.get("vector_attempted")) is not bool:
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
        vector_attempted=False if unversioned else obj["vector_attempted"],
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


def _read_record(path: Path, *, allow_unversioned_terminal: bool = False) -> MutationRecord:
    data: Any = None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Record payload is not a JSON object.")
        record = _record_from_dict(data, allow_unversioned_terminal=allow_unversioned_terminal)
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


def _disappeared_during_read(exc: MutationRecoveryRequiredError) -> bool:
    return isinstance(exc.__cause__, FileNotFoundError)


def _json_paths(directory: Path) -> list[Path]:
    try:
        with os.scandir(directory) as entries:
            paths: list[Path] = []
            for entry in entries:
                if not entry.name.endswith(".json"):
                    continue
                if not entry.is_file(follow_symlinks=False):
                    raise MutationRecoveryRequiredError(
                        f"Mutation journal record at {entry.path} is not a regular file."
                    )
                paths.append(Path(entry.path))
            return sorted(paths)
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise MutationRecoveryRequiredError(
            f"Cannot enumerate mutation journal records at {directory}: {exc}"
        ) from exc


class FileMutationJournal(MutationJournalPort):
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.active_dir = self.root / "active"
        self.done_dir = self.root / "done"
        self._legacy_checked = False
        self._legacy_incomplete_op_ids: tuple[str, ...] = ()
        self._legacy_overlay_paths: frozenset[Path] = frozenset()
        self._last_sweep_monotonic = float("-inf")
        self._validated_done: dict[Path, tuple[int, int, int, int]] = {}

    def _check_legacy_once(self) -> None:
        if self._legacy_checked:
            return
        incomplete_op_ids: list[str] = []
        for path in _json_paths(self.root):
            record = _read_record(path, allow_unversioned_terminal=True)
            if record.state not in _TERMINAL_STATES:
                incomplete_op_ids.append(record.op_id)
        self._legacy_incomplete_op_ids = tuple(incomplete_op_ids)
        self._legacy_overlay_paths = frozenset(
            _record_path(self.done_dir, op_id) for op_id in incomplete_op_ids
        )
        self._legacy_checked = True

    def _unresolved_legacy_records(self, *, seen_op_ids: set[str]) -> Iterator[MutationRecord]:
        for op_id in self._legacy_incomplete_op_ids:
            if op_id in seen_op_ids:
                continue
            record = self.get(op_id)
            if record is None:
                raise MutationRecoveryRequiredError(
                    f"Legacy mutation journal record for op_id={op_id!r} vanished without a receipt."
                )
            if record.state not in _TERMINAL_STATES:
                yield record

    def _done_files(self) -> list[tuple[Path, os.stat_result]]:
        files: list[tuple[Path, os.stat_result]] = []
        for path in _json_paths(self.done_dir):
            try:
                info = path.stat()
            except FileNotFoundError:
                # Retention can remove a receipt after directory enumeration.
                continue
            except OSError as exc:
                raise MutationRecoveryRequiredError(
                    f"Cannot inspect mutation journal receipt at {path}: {exc}"
                ) from exc
            signature = (info.st_ino, info.st_mtime_ns, info.st_size, info.st_mode)
            if self._validated_done.get(path) != signature:
                try:
                    record = _read_record(path)
                except MutationRecoveryRequiredError as exc:
                    if _disappeared_during_read(exc):
                        continue
                    raise
                if record.state not in _TERMINAL_STATES:
                    raise MutationRecoveryRequiredError(
                        f"Non-terminal mutation journal receipt at {path} "
                        f"op_id={record.op_id!r} state={record.state!r}."
                    )
                self._validated_done[path] = signature
            files.append((path, info))
        present = {path for path, _ in files}
        self._validated_done = {
            path: signature for path, signature in self._validated_done.items() if path in present
        }
        return files

    def _sweep_done(self) -> None:
        cutoff = time.time() - _DONE_MAX_AGE_S
        retained: list[tuple[Path, os.stat_result]] = []
        removed = False
        for path, info in self._done_files():
            if path not in self._legacy_overlay_paths and info.st_mtime < cutoff:
                path.unlink()
                removed = True
                self._validated_done.pop(path, None)
            else:
                retained.append((path, info))
        total_bytes = sum(info.st_size for _, info in retained)
        for path, info in sorted(retained, key=lambda item: (item[1].st_mtime_ns, item[0].name)):
            if total_bytes <= _DONE_MAX_BYTES:
                break
            if path in self._legacy_overlay_paths:
                continue
            path.unlink()
            removed = True
            self._validated_done.pop(path, None)
            total_bytes -= info.st_size
        if removed:
            fsync_directory(self.done_dir)

    def _maybe_sweep_done(self) -> None:
        now = time.monotonic()
        if now - self._last_sweep_monotonic < _DONE_SWEEP_INTERVAL_S:
            return
        try:
            self._sweep_done()
        except OSError as exc:
            raise MutationRecoveryRequiredError(
                f"Cannot prune completed mutation journal receipts at {self.done_dir}: {exc}"
            ) from exc
        self._last_sweep_monotonic = now

    def _move_to_done(self, path: Path, op_id: str) -> None:
        try:
            ensure_durable_directory(self.done_dir)
            os.replace(path, _record_path(self.done_dir, op_id))
            # Persist the new receipt before persisting removal from active/.
            fsync_directory(self.done_dir)
            fsync_directory(self.active_dir)
        except OSError as exc:
            raise MutationRecoveryRequiredError(
                f"Cannot finish mutation journal record at {path} op_id={op_id!r}: {exc}"
            ) from exc

    def get(self, op_id: str) -> MutationRecord | None:
        observed_active_vanish = False
        for attempt in range(2):
            for directory in (self.active_dir, self.done_dir, self.root):
                path = _record_path(directory, op_id)
                if path.is_symlink():
                    raise MutationRecoveryRequiredError(
                        f"Mutation journal record at {path} is not a regular file."
                    )
                if path.is_file():
                    try:
                        return _read_record(path, allow_unversioned_terminal=directory == self.root)
                    except MutationRecoveryRequiredError as exc:
                        if not _disappeared_during_read(exc):
                            raise
                        observed_active_vanish |= directory == self.active_dir
                        continue
                if path.exists():
                    raise MutationRecoveryRequiredError(
                        f"Mutation journal record at {path} is not a regular file."
                    )
            if not observed_active_vanish:
                return None
            if attempt == 0:
                continue
        raise MutationRecoveryRequiredError(
            f"Active mutation journal record for op_id={op_id!r} vanished without a receipt."
        )

    def upsert(self, record: MutationRecord) -> None:
        self._check_legacy_once()
        now = time.time()
        existing = self.get(record.op_id)
        normalized = MutationRecord(
            op_id=record.op_id,
            state=record.state,
            intent=dict(record.intent),
            before_image=(
                None
                if record.state == "COMMITTED"
                else dict(record.before_image)
                if record.before_image is not None
                else None
            ),
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
        path = _record_path(self.active_dir, normalized.op_id)
        try:
            atomic_write_text(
                path,
                json.dumps(_record_to_dict(normalized), ensure_ascii=True, separators=(",", ":")),
            )
            if normalized.state in _TERMINAL_STATES:
                self._move_to_done(path, normalized.op_id)
        except OSError as exc:
            raise MutationRecoveryRequiredError(
                f"Cannot persist mutation journal at {path} "
                f"op_id={record.op_id!r} state={record.state!r}: {exc}"
            ) from exc

    def list_incomplete(self, *, limit: int = 100) -> list[MutationRecord]:
        if limit <= 0:
            raise ValueError("Mutation recovery limit must be positive.")
        self._check_legacy_once()
        self._maybe_sweep_done()
        out: list[MutationRecord] = []
        seen_op_ids: set[str] = set()
        for path in _json_paths(self.active_dir):
            record = _read_record(path)
            if record.state in _TERMINAL_STATES:
                self._move_to_done(path, record.op_id)
                continue
            out.append(record)
            seen_op_ids.add(record.op_id)
            if len(out) >= limit:
                break
        if len(out) < limit:
            for record in self._unresolved_legacy_records(seen_op_ids=seen_op_ids):
                out.append(record)
                if len(out) >= limit:
                    break
        return out

    def count_incomplete(self) -> int:
        """Inspect readiness without changing journal files or running retention."""
        self._check_legacy_once()
        self._done_files()
        active_op_ids: set[str] = set()
        for path in _json_paths(self.active_dir):
            try:
                record = _read_record(path)
            except MutationRecoveryRequiredError as exc:
                if not _disappeared_during_read(exc):
                    raise
                done_path = self.done_dir / path.name
                if done_path.is_file() and _read_record(done_path).state in _TERMINAL_STATES:
                    continue
                raise
            if record.state not in _TERMINAL_STATES:
                active_op_ids.add(record.op_id)
        return len(active_op_ids) + sum(
            1 for _ in self._unresolved_legacy_records(seen_op_ids=active_op_ids)
        )


__all__ = ["FileMutationJournal"]

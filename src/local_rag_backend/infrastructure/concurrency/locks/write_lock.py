"""
Cross-process lock for multi-store write operations.

Why:
- SQL + vector index updates are not transactional across both stores.
- Concurrent writers can interleave commits and leave SQL/vector drift.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING

from local_rag_backend.infrastructure.concurrency.locks.file_lock import exclusive_file_lock
from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_LOCAL_WRITE_LOCK = threading.RLock()
_THREAD_STATE = threading.local()


def _exclusive_file_lock(
    lock_path: Path, *, timeout_s: float = 30.0, poll_s: float = 0.05
) -> AbstractContextManager[None]:
    return exclusive_file_lock(
        lock_path,
        error_message=(
            f"Unable to acquire multi-store write lock at {lock_path}. "
            "Refusing to run mutating operation without a cross-process lock."
        ),
        timeout_s=timeout_s,
        poll_s=poll_s,
    )


def _record_lock_event(
    *,
    lock_path: Path,
    status: str,
    wait_s: float | None = None,
    hold_s: float | None = None,
) -> None:
    metrics_path = str(getattr(settings, "lock_metrics_path", "") or "").strip()
    if not metrics_path:
        return
    payload: dict[str, float | int | str] = {
        "ts": time.time(),
        "pid": os.getpid(),
        "lock_path": str(lock_path),
        "status": str(status),
    }
    if wait_s is not None:
        payload["wait_ms"] = max(0.0, float(wait_s)) * 1000.0
    if hold_s is not None:
        payload["hold_ms"] = max(0.0, float(hold_s)) * 1000.0
    try:
        with open(metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")
    except Exception:
        # Best-effort metrics hook for load/stress testing. Never break writes.
        return


@contextmanager
def multi_store_write_lock(
    *,
    coordination_dir: Path | None = None,
    timeout_s: float | None = None,
    poll_s: float | None = None,
) -> Iterator[None]:
    """
    Serialize mutating multi-store operations (SQL + FAISS) across threads/processes.
    """
    depth = int(getattr(_THREAD_STATE, "depth", 0))
    if depth > 0:
        _THREAD_STATE.depth = depth + 1
        try:
            yield
        finally:
            _THREAD_STATE.depth = max(0, int(getattr(_THREAD_STATE, "depth", 1)) - 1)
        return

    lock_root = coordination_dir or settings.get_coordination_dir()
    lock_path = lock_root / ".rag_multi_store_write.lock"
    resolved_timeout = (
        float(timeout_s)
        if timeout_s is not None
        else float(getattr(settings, "write_lock_timeout_s", 30.0))
    )
    resolved_poll = (
        float(poll_s) if poll_s is not None else float(getattr(settings, "write_lock_poll_s", 0.05))
    )
    file_lock_cm = _exclusive_file_lock(lock_path, timeout_s=resolved_timeout, poll_s=resolved_poll)
    acquire_started = time.monotonic()
    try:
        with _LOCAL_WRITE_LOCK, file_lock_cm:
            acquired_wait_s = time.monotonic() - acquire_started
            _record_lock_event(
                lock_path=lock_path,
                status="acquired",
                wait_s=acquired_wait_s,
            )
            hold_started = time.monotonic()
            _THREAD_STATE.depth = 1
            try:
                yield
            finally:
                _THREAD_STATE.depth = 0
                _record_lock_event(
                    lock_path=lock_path,
                    status="released",
                    hold_s=(time.monotonic() - hold_started),
                )
    except Exception:
        _record_lock_event(
            lock_path=lock_path,
            wait_s=(time.monotonic() - acquire_started),
            status="failed",
        )
        raise


__all__ = ["_exclusive_file_lock", "multi_store_write_lock"]

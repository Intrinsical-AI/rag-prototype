"""
Cross-process lock for multi-store write operations.

Why:
- SQL + vector index updates are not transactional across both stores.
- Concurrent writers can interleave commits and leave SQL/vector drift.
"""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING

from local_rag_backend.core.services.file_lock import exclusive_file_lock
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
    try:
        file_lock_cm = _exclusive_file_lock(
            lock_path, timeout_s=resolved_timeout, poll_s=resolved_poll
        )
    except TypeError:
        # Compatibility for tests/overrides patching `_exclusive_file_lock(path)`.
        file_lock_cm = _exclusive_file_lock(lock_path)
    with _LOCAL_WRITE_LOCK, file_lock_cm:
        _THREAD_STATE.depth = 1
        try:
            yield
        finally:
            _THREAD_STATE.depth = 0

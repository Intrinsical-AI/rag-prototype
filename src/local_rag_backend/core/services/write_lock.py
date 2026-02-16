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


def _exclusive_file_lock(lock_path: Path) -> AbstractContextManager[None]:
    return exclusive_file_lock(
        lock_path,
        error_message=(
            f"Unable to acquire multi-store write lock at {lock_path}. "
            "Refusing to run mutating operation without a cross-process lock."
        ),
    )


@contextmanager
def multi_store_write_lock(*, coordination_dir: Path | None = None) -> Iterator[None]:
    """
    Serialize mutating multi-store operations (SQL + FAISS) across threads/processes.
    """
    lock_root = coordination_dir or settings.get_coordination_dir()
    lock_path = lock_root / ".rag_multi_store_write.lock"
    with _LOCAL_WRITE_LOCK, _exclusive_file_lock(lock_path):
        yield

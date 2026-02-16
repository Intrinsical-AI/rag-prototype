"""
Cross-process lock for multi-store write operations.

Why:
- SQL + vector index updates are not transactional across both stores.
- Concurrent writers can interleave commits and leave SQL/vector drift.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any

from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_LOCAL_WRITE_LOCK = threading.RLock()


@contextmanager
def _exclusive_file_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = lock_path.open("a+b")
    locked = False
    try:
        try:  # POSIX
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            locked = True
        except Exception:  # pragma: no cover
            locked = False

        if not locked:  # pragma: no cover
            try:  # Windows
                import msvcrt  # pragma: no cover

                msvcrt_any: Any = msvcrt  # pragma: no cover
                f.seek(0, os.SEEK_END)  # pragma: no cover
                if f.tell() == 0:  # pragma: no cover
                    f.write(b"0")  # pragma: no cover
                    f.flush()  # pragma: no cover
                f.seek(0)  # pragma: no cover
                msvcrt_any.locking(  # pragma: no cover
                    f.fileno(), getattr(msvcrt_any, "LK_LOCK", 1), 1
                )
                locked = True  # pragma: no cover
            except Exception:  # pragma: no cover
                locked = False  # pragma: no cover

        if not locked:
            raise RuntimeError(
                f"Unable to acquire multi-store write lock at {lock_path}. "
                "Refusing to run mutating operation without a cross-process lock."
            )

        yield
    finally:
        if locked:
            with suppress(Exception):  # pragma: no cover
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            with suppress(Exception):  # pragma: no cover
                import msvcrt  # pragma: no cover

                msvcrt_mod: Any = msvcrt  # pragma: no cover
                f.seek(0)  # pragma: no cover
                msvcrt_mod.locking(  # pragma: no cover
                    f.fileno(), getattr(msvcrt_mod, "LK_UNLCK", 0), 1
                )
        f.close()


@contextmanager
def multi_store_write_lock() -> Iterator[None]:
    """
    Serialize mutating multi-store operations (SQL + FAISS) across threads/processes.
    """
    lock_path = settings.get_coordination_dir() / ".rag_multi_store_write.lock"
    with _LOCAL_WRITE_LOCK, _exclusive_file_lock(lock_path):
        yield

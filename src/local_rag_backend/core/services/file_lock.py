"""
Shared cross-process exclusive file locking (stdlib-only).
"""

from __future__ import annotations

import os
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _try_posix_lock(file_obj: Any) -> bool:
    try:  # POSIX
        import fcntl

        fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
        return True
    except Exception:  # pragma: no cover
        return False


def _try_windows_lock(file_obj: Any) -> bool:  # pragma: no cover
    try:
        import msvcrt

        msvcrt_any: Any = msvcrt
        file_obj.seek(0, os.SEEK_END)
        if file_obj.tell() == 0:
            file_obj.write(b"0")
            file_obj.flush()
        file_obj.seek(0)
        msvcrt_any.locking(file_obj.fileno(), getattr(msvcrt_any, "LK_LOCK", 1), 1)
        return True
    except Exception:
        return False


def _best_effort_unlock(file_obj: Any) -> None:
    with suppress(Exception):  # pragma: no cover
        import fcntl

        fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
    with suppress(Exception):  # pragma: no cover
        import msvcrt  # pragma: no cover

        msvcrt_mod: Any = msvcrt  # pragma: no cover
        file_obj.seek(0)  # pragma: no cover
        msvcrt_mod.locking(  # pragma: no cover
            file_obj.fileno(), getattr(msvcrt_mod, "LK_UNLCK", 0), 1
        )


@contextmanager
def exclusive_file_lock(lock_path: Path, *, error_message: str) -> Iterator[None]:
    """
    Cross-process exclusive lock using only stdlib.

    - POSIX: fcntl.flock
    - Windows: msvcrt.locking
    - Else: fail-closed
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = lock_path.open("a+b")
    locked = False
    try:
        locked = _try_posix_lock(f)
        if not locked:
            locked = _try_windows_lock(f)

        if not locked:
            raise RuntimeError(error_message)

        yield
    finally:
        if locked:
            _best_effort_unlock(f)
        f.close()

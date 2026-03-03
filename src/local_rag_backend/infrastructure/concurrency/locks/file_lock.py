"""
Shared cross-process exclusive file locking (stdlib-only).
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any

from local_rag_backend.core.errors import WriteLockTimeoutError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _try_posix_lock(file_obj: Any, *, non_blocking: bool) -> bool | None:
    try:  # POSIX
        import fcntl

        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if non_blocking else 0)
        fcntl.flock(file_obj.fileno(), flags)
        return True
    except BlockingIOError:
        return False
    except Exception:  # pragma: no cover
        return None


def _try_windows_lock(file_obj: Any, *, non_blocking: bool) -> bool | None:  # pragma: no cover
    try:
        import msvcrt

        msvcrt_any: Any = msvcrt
        file_obj.seek(0, os.SEEK_END)
        if file_obj.tell() == 0:
            file_obj.write(b"0")
            file_obj.flush()
        file_obj.seek(0)
        lock_mode = (
            getattr(msvcrt_any, "LK_NBLCK", 1)
            if non_blocking
            else getattr(msvcrt_any, "LK_LOCK", 1)
        )
        msvcrt_any.locking(file_obj.fileno(), lock_mode, 1)
        return True
    except OSError:
        return None
    except Exception:
        return None


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
def exclusive_file_lock(
    lock_path: Path,
    *,
    error_message: str,
    timeout_s: float = 30.0,
    poll_s: float = 0.05,
) -> Iterator[None]:
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
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            posix = _try_posix_lock(f, non_blocking=True)
            windows = _try_windows_lock(f, non_blocking=True) if posix is not True else None

            if posix is True or windows is True:
                locked = True
                break

            # Fail closed if the runtime cannot acquire a lock on this platform.
            if posix is None and windows is None:
                raise RuntimeError(error_message)

            now = time.monotonic()
            if now >= deadline:
                raise WriteLockTimeoutError(
                    f"{error_message} Timed out after {float(timeout_s):.2f}s."
                )
            time.sleep(max(0.001, float(poll_s)))

        yield
    finally:
        if locked:
            _best_effort_unlock(f)
        f.close()


__all__ = ["exclusive_file_lock"]

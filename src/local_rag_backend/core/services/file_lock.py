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
            raise RuntimeError(error_message)

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

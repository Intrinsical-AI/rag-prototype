from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from pathlib import Path


def fsync_directory(path: Path) -> None:
    """Persist directory entries after a create, rename, or deletion."""
    if os.name == "nt":
        # Python exposes no portable directory fsync on Windows.
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ensure_durable_directory(path: Path) -> None:
    """Create missing ancestors and persist each new directory entry."""
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    if not current.is_dir():
        raise NotADirectoryError(current)
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        fsync_directory(directory.parent)


def atomic_replace_file(temp_path: Path, target_path: Path) -> None:
    """Publish an already-written file only after its bytes are durable."""
    ensure_durable_directory(target_path.parent)
    with temp_path.open("rb+") as temp_file:
        os.fsync(temp_file.fileno())
    os.replace(temp_path, target_path)
    fsync_directory(target_path.parent)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    ensure_durable_directory(path.parent)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False
    ) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = Path(tmp.name)
    try:
        atomic_replace_file(tmp_path, path)
    finally:
        with suppress(OSError):
            tmp_path.unlink()


def atomic_write_text(path: Path, content: str) -> None:
    atomic_write_bytes(path, content.encode("utf-8"))

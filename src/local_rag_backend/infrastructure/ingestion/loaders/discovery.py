# src/local_rag_backend/infrastructure/ingestion/loaders/discovery.py
"""
Path/dir discovery helpers for ingestion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path


def discover_files(
    inputs: Sequence[Path],
    *,
    recursive: bool,
    follow_symlinks: bool,
    max_files: int,
    max_file_bytes: int,
    max_total_bytes: int,
) -> Iterable[Path]:
    """
    Yield files from a mix of files/directories under basic safety limits.

    Notes:
    - Does not read file contents (limits are based on stat size).
    - Binary/text detection happens later (factory).
    """
    max_files_i = max(0, int(max_files))
    max_file_bytes_i = max(0, int(max_file_bytes))
    max_total_bytes_i = max(0, int(max_total_bytes))

    seen = 0
    total = 0

    def _yield_file(p: Path) -> Iterable[Path]:
        nonlocal seen, total
        if max_files_i and seen >= max_files_i:
            return []
        try:
            st = p.stat()
        except Exception:
            return []

        if not follow_symlinks and p.is_symlink():
            return []
        if not p.is_file():
            return []
        size = int(getattr(st, "st_size", 0) or 0)
        if max_file_bytes_i and size > max_file_bytes_i:
            return []
        if max_total_bytes_i and (total + size) > max_total_bytes_i:
            return []

        seen += 1
        total += size
        return [p]

    for inp in inputs:
        if not follow_symlinks and inp.is_symlink():
            continue

        if inp.is_file():
            yield from _yield_file(inp)
            continue

        if inp.is_dir():
            it = inp.rglob("*") if recursive else inp.glob("*")
            for p in it:
                yield from _yield_file(p)

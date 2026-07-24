from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path


def _configured_path(
    variable: str,
    *,
    predicate: Callable[[Path], bool],
    expected_kind: str,
) -> Path | None:
    value = os.environ.get(variable)
    if not value:
        return None
    path = Path(value).expanduser()
    if not predicate(path):
        raise RuntimeError(f"{variable} must point to an existing {expected_kind}: {path}")
    return path


def configured_file(variable: str) -> Path | None:
    return _configured_path(variable, predicate=Path.is_file, expected_kind="file")


def configured_directory(variable: str) -> Path | None:
    return _configured_path(variable, predicate=Path.is_dir, expected_kind="directory")

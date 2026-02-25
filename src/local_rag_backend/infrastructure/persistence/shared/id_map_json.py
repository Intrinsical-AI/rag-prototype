from __future__ import annotations

import json
from pathlib import Path

from local_rag_backend.infrastructure.persistence.shared.atomic_io import atomic_write_text


def looks_like_pickle(data: bytes) -> bool:
    # Pickle protocol v2+ starts with 0x80 <protocol>.
    return data.startswith(b"\x80")


def load_id_map_json(path: Path) -> list[int]:
    if not path.exists():
        return []

    raw = path.read_bytes()
    if not raw:
        raise ValueError("Invalid id_map format: empty file.")
    if looks_like_pickle(raw):
        raise RuntimeError(
            f"Unsafe pickle id_map detected at {path}. "
            "Delete it and rebuild the index (or migrate it to JSON)."
        )

    try:
        loaded = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Invalid id_map format at {path}: expected JSON list[int].") from e

    if not isinstance(loaded, list) or not all(isinstance(x, int) for x in loaded):
        raise ValueError("Invalid id_map format: expected a JSON list[int].")
    return loaded


def save_id_map_json(path: Path, id_map: list[int]) -> None:
    atomic_write_text(path, json.dumps(list(id_map)))

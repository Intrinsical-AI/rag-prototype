"""
Index manifest persistence and drift detection.

We persist a small JSON manifest next to the FAISS index to detect configuration drift:
- embedding backend/model
- embedding dimension
- chunker strategy/version

This avoids silent index corruption (e.g. writing 384-d vectors into a 1536-d index)
and provides actionable diagnostics in /ready and `rag-status`.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from local_rag_backend.settings import Settings

MANIFEST_VERSION = 1


def manifest_path_for(index_path: str | Path) -> Path:
    """
    Compute the manifest path for an index file.

    Contract: sibling file named `index_manifest.json` in the same directory as the index.
    """
    idx = Path(index_path)
    return idx.with_name("index_manifest.json")


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def read_manifest(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    raw = p.read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError("Empty manifest file.")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Invalid manifest format: expected JSON object.")
    return data


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    p = Path(path)
    _atomic_write_text(p, json.dumps(manifest, sort_keys=True))


def build_expected_manifest_config(
    *,
    embedding_backend: str,
    embedding_model: str,
    chunker_strategy: str,
    chunker_version: str,
) -> dict[str, str]:
    # Keep this intentionally small: it's used by diagnostics to compare stable identifiers.
    return {
        "embedding_backend": str(embedding_backend),
        "embedding_model": str(embedding_model),
        "chunker_strategy": str(chunker_strategy),
        "chunker_version": str(chunker_version),
    }


def expected_manifest_config_from_settings(cfg: Settings = settings) -> dict[str, str]:
    embedding_backend = "openai" if bool(cfg.openai_api_key) else "sentence_transformers"
    embedding_model = (
        cfg.openai_embedding_model if bool(cfg.openai_api_key) else cfg.st_embedding_model
    )
    return build_expected_manifest_config(
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        chunker_strategy=cfg.ingest_chunk_strategy,
        chunker_version=cfg.ingest_chunker_version,
    )


def build_manifest(
    *,
    expected: dict[str, str],
    dimension: int,
    index_backend: str,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    now = _utc_now_iso()
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": created_at or now,
        "updated_at": updated_at or now,
        "dimension": int(dimension),
        "index_backend": str(index_backend),
        **expected,
    }


def create_manifest_if_missing_for_settings(
    *,
    index_path: str | Path,
    expected: dict[str, str],
    dimension: int,
    index_backend: str,
) -> Path:
    """
    Create the manifest if missing (do not overwrite an existing manifest).

    Behavior:
    - If missing: create it with the provided expected config.
    - If present: no-op.
    """
    mpath = manifest_path_for(index_path)
    existing = read_manifest(mpath)
    if existing is None:
        manifest = build_manifest(
            expected=expected,
            dimension=dimension,
            index_backend=index_backend,
            created_at=_utc_now_iso(),
            updated_at=_utc_now_iso(),
        )
        write_manifest(mpath, manifest)
    return mpath


def overwrite_manifest_for_settings(
    *,
    index_path: str | Path,
    expected: dict[str, str],
    dimension: int,
    index_backend: str,
) -> Path:
    """
    Overwrite the manifest to match the provided expected config.

    This is intended for full rebuilds: it updates config identifiers and `updated_at`,
    while preserving `created_at` if present.
    """
    mpath = manifest_path_for(index_path)
    existing = read_manifest(mpath)
    created_at = None
    if isinstance(existing, dict):
        ca = existing.get("created_at")
        if isinstance(ca, str) and ca.strip():
            created_at = ca

    manifest = build_manifest(
        expected=expected,
        dimension=dimension,
        index_backend=index_backend,
        created_at=created_at,
        updated_at=_utc_now_iso(),
    )
    write_manifest(mpath, manifest)
    return mpath


@dataclass(frozen=True)
class ManifestMismatch:
    key: str
    expected: Any
    actual: Any


def validate_manifest(
    *,
    manifest: dict[str, Any],
    expected_config: dict[str, str] | None,
    actual_dimension: int,
    actual_index_backend: str,
) -> tuple[list[ManifestMismatch], list[str]]:
    """
    Validate a manifest and return:
    - mismatches: structured key-level mismatches (drift)
    - errors: non-drift invalidities (corruption)
    """
    mismatches: list[ManifestMismatch] = []
    errors: list[str] = []

    version = manifest.get("manifest_version")
    if version != MANIFEST_VERSION:
        errors.append(f"Unsupported manifest_version={version!r} (expected {MANIFEST_VERSION}).")

    dim = manifest.get("dimension")
    if not isinstance(dim, int):
        errors.append("Manifest dimension missing/invalid.")
    elif int(dim) != int(actual_dimension):
        errors.append(
            f"Manifest dimension {int(dim)} does not match index dimension {int(actual_dimension)}."
        )

    ib = manifest.get("index_backend")
    if isinstance(ib, str) and ib and ib != str(actual_index_backend):
        # Backend mismatch is usually a packaging/runtime difference (faiss vs numpy fallback).
        # It's not necessarily fatal, but it helps explain behavior. Report as drift.
        mismatches.append(
            ManifestMismatch(key="index_backend", expected=str(actual_index_backend), actual=ib)
        )

    if expected_config:
        for k, v in expected_config.items():
            actual = manifest.get(k)
            if actual != v:
                mismatches.append(ManifestMismatch(key=k, expected=v, actual=actual))

    return mismatches, errors


def purge_index_artifacts(*, index_path: str | Path, id_map_path: str | Path) -> None:
    """
    Remove index-related files so a rebuild can recreate them cleanly.

    This is used to recover from situations where the on-disk index has an incompatible
    dimension/backend and cannot be loaded for an in-place rebuild.
    """
    idx = Path(index_path)
    idmap = Path(id_map_path)
    lock = idx.with_name(idx.name + ".lock")
    manifest = manifest_path_for(idx)
    for p in (idx, idmap, manifest, lock):
        with suppress(FileNotFoundError):
            p.unlink()

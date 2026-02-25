"""
Index manifest persistence and drift detection.

We persist a small JSON manifest next to the vector index to detect configuration drift:
- embedding backend/model
- embedding dimension
- chunker strategy/version
"""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_rag_backend.infrastructure.persistence.shared.atomic_io import atomic_write_text
from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from local_rag_backend.settings import Settings

MANIFEST_VERSION = 1


def manifest_path_for(index_path: str | Path) -> Path:
    idx = Path(index_path)
    return idx.with_name("index_manifest.json")


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


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
    atomic_write_text(p, json.dumps(manifest, sort_keys=True))


def build_expected_manifest_config(
    *,
    embedding_backend: str,
    embedding_model: str,
    chunker_strategy: str,
    chunker_version: str,
) -> dict[str, str]:
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
    idx = Path(index_path)
    idmap = Path(id_map_path)
    lock = idx.with_name(idx.name + ".lock")
    manifest = manifest_path_for(idx)
    for p in (idx, idmap, manifest, lock):
        with suppress(FileNotFoundError):
            p.unlink()

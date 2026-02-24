"""
Operational diagnostics for the app layer (health/readiness/status).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


def get_documents_count(engine: Engine) -> int:
    with engine.connect() as conn:
        return int(conn.execute(text("SELECT COUNT(*) FROM documents")).scalar() or 0)


def get_history_count(engine: Engine) -> int:
    with engine.connect() as conn:
        return int(conn.execute(text("SELECT COUNT(*) FROM qa_history")).scalar() or 0)


def get_document_ids(engine: Engine, *, limit: int | None = None) -> list[int]:
    sql = "SELECT id FROM documents ORDER BY id"
    if limit is not None:
        sql += " LIMIT :limit"
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"limit": limit} if limit is not None else {}).fetchall()
    return [int(r[0]) for r in rows]


def _build_missing_index_stats(
    *, idx_path: Path, map_path: Path, manifest_path: Path
) -> dict[str, Any] | None:
    missing: list[str] = []
    if not idx_path.exists():
        missing.append(str(idx_path))
    if not map_path.exists():
        missing.append(str(map_path))
    if not missing:
        return None
    return {
        "status": "missing",
        "missing": missing,
        "index_path": str(idx_path),
        "id_map_path": str(map_path),
        "manifest_path": str(manifest_path),
        "hint": "Create/rebuild the index (e.g. `rag-rebuild-index` or POST /api/index/rebuild).",
    }


def _build_ok_index_payload(
    *,
    idx_path: Path,
    map_path: Path,
    manifest_path: Path,
    backend: str,
    dim: int,
    vectors: int,
    id_map_len: int,
    unique_ids: int,
    duplicates: int,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "index_path": str(idx_path),
        "id_map_path": str(map_path),
        "manifest_path": str(manifest_path),
        "backend": backend,
        "dim": dim,
        "vectors": vectors,
        "id_map_len": id_map_len,
        "unique_ids": unique_ids,
        "duplicates": duplicates,
    }


def _apply_vector_integrity_status(
    payload: dict[str, Any], *, vectors: int, id_map_len: int
) -> None:
    duplicates = int(payload.get("duplicates") or 0)
    if vectors != id_map_len:
        payload["status"] = "corrupt"
        payload["error"] = "index/id_map length mismatch"
        payload["hint"] = "Rebuild the index to repair drift (e.g. `rag-rebuild-index`)."
        return
    if duplicates:
        payload["status"] = "corrupt"
        payload["error"] = "duplicate document IDs in id_map"
        payload["hint"] = "Rebuild the index to remove duplicates (e.g. `rag-rebuild-index`)."


def _apply_manifest_status(
    *,
    payload: dict[str, Any],
    manifest_path: Path,
    expected_manifest: dict[str, str] | None,
    actual_dimension: int,
    actual_index_backend: str,
) -> None:
    from local_rag_backend.infrastructure.persistence.faiss.manifest import (
        read_manifest,
        validate_manifest,
    )

    try:
        manifest = read_manifest(manifest_path)
        payload["manifest_present"] = bool(manifest)
        payload["manifest"] = manifest

        if payload["status"] != "ok":
            return
        if manifest is None:
            payload["status"] = "drift"
            payload["error"] = "index manifest missing"
            payload["hint"] = (
                "Rebuild the index to generate a manifest and prevent config drift "
                "(e.g. `rag-rebuild-index`)."
            )
            return

        mismatches, errors = validate_manifest(
            manifest=manifest,
            expected_config=expected_manifest,
            actual_dimension=actual_dimension,
            actual_index_backend=actual_index_backend,
        )
        payload["manifest_errors"] = errors
        payload["manifest_mismatches"] = [
            {"key": m.key, "expected": m.expected, "actual": m.actual} for m in mismatches
        ]
        if errors:
            payload["status"] = "corrupt"
            payload["error"] = "invalid index manifest"
            payload["hint"] = "Rebuild the index to repair the manifest (e.g. `rag-rebuild-index`)."
            return
        if mismatches:
            payload["status"] = "drift"
            payload["error"] = "index manifest mismatch"
            payload["hint"] = "Rebuild the index to remove drift (e.g. `rag-rebuild-index`)."
    except Exception as e:
        if payload["status"] == "ok":
            payload["status"] = "corrupt"
            payload["error"] = f"manifest read/validation failed: {type(e).__name__}: {e}"
            payload["hint"] = "Rebuild the index to repair the manifest (e.g. `rag-rebuild-index`)."


def get_retrieval_index_stats(
    *,
    index_path: str | Path,
    id_map_path: str | Path,
    dim: int | None = None,
    expected_manifest: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Return best-effort stats for the on-disk retrieval index.

    This is safe to call even when optional deps are missing. It returns:
      - status: ok|missing|corrupt|drift
      - hint: remediation hint when not ok
      - vectors/id_map_len/unique_ids/duplicates/dim/backend when ok (best-effort)
    """
    idx_path = Path(index_path)
    map_path = Path(id_map_path)
    from local_rag_backend.infrastructure.persistence.faiss.manifest import manifest_path_for

    manifest_path = manifest_path_for(idx_path)

    if missing_payload := _build_missing_index_stats(
        idx_path=idx_path, map_path=map_path, manifest_path=manifest_path
    ):
        return missing_payload

    try:
        from local_rag_backend.infrastructure.persistence.faiss.index import FaissIndex

        idx = FaissIndex(idx_path, map_path, dim=dim)
        id_map_len = len(idx.id_map)
        unique_ids = len(set(idx.id_map))
        duplicates = int(id_map_len - unique_ids)
        vectors = int(idx.ntotal)
        backend = str(idx.backend)

        payload = _build_ok_index_payload(
            idx_path=idx_path,
            map_path=map_path,
            manifest_path=manifest_path,
            backend=backend,
            dim=int(idx.dim),
            vectors=vectors,
            id_map_len=id_map_len,
            unique_ids=unique_ids,
            duplicates=duplicates,
        )
        _apply_vector_integrity_status(payload, vectors=vectors, id_map_len=id_map_len)
        _apply_manifest_status(
            payload=payload,
            manifest_path=manifest_path,
            expected_manifest=expected_manifest,
            actual_dimension=int(idx.dim),
            actual_index_backend=backend,
        )

        return payload
    except Exception as e:
        return {
            "status": "corrupt",
            "index_path": str(idx_path),
            "id_map_path": str(map_path),
            "manifest_path": str(manifest_path),
            "error": f"{type(e).__name__}: {e}",
            "hint": "Rebuild the index (e.g. `rag-rebuild-index` or POST /api/index/rebuild).",
        }

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
    from local_rag_backend.infrastructure.persistence.faiss.manifest import (
        manifest_path_for,
        read_manifest,
        validate_manifest,
    )

    manifest_path = manifest_path_for(idx_path)

    missing: list[str] = []
    if not idx_path.exists():
        missing.append(str(idx_path))
    if not map_path.exists():
        missing.append(str(map_path))
    if missing:
        return {
            "status": "missing",
            "missing": missing,
            "index_path": str(idx_path),
            "id_map_path": str(map_path),
            "manifest_path": str(manifest_path),
            "hint": "Create/rebuild the index (e.g. `rag-rebuild-index` or POST /api/index/rebuild).",
        }

    try:
        from local_rag_backend.infrastructure.persistence.faiss.index import FaissIndex

        idx = FaissIndex(idx_path, map_path, dim=dim)
        id_map_len = len(idx.id_map)
        unique_ids = len(set(idx.id_map))
        duplicates = int(id_map_len - unique_ids)
        vectors = int(idx.ntotal)
        backend = str(idx.backend)

        payload: dict[str, Any] = {
            "status": "ok",
            "index_path": str(idx_path),
            "id_map_path": str(map_path),
            "manifest_path": str(manifest_path),
            "backend": backend,
            "dim": int(idx.dim),
            "vectors": vectors,
            "id_map_len": id_map_len,
            "unique_ids": unique_ids,
            "duplicates": duplicates,
        }

        if vectors != id_map_len:
            payload["status"] = "corrupt"
            payload["error"] = "index/id_map length mismatch"
            payload["hint"] = "Rebuild the index to repair drift (e.g. `rag-rebuild-index`)."
        elif duplicates:
            payload["status"] = "corrupt"
            payload["error"] = "duplicate document IDs in id_map"
            payload["hint"] = "Rebuild the index to remove duplicates (e.g. `rag-rebuild-index`)."

        # Manifest checks (drift detection against stable identifiers).
        # Only override status when the underlying index/id_map is consistent.
        try:
            manifest = read_manifest(manifest_path)
            payload["manifest_present"] = bool(manifest)
            payload["manifest"] = manifest

            if payload["status"] == "ok":
                if manifest is None:
                    payload["status"] = "drift"
                    payload["error"] = "index manifest missing"
                    payload["hint"] = (
                        "Rebuild the index to generate a manifest and prevent config drift "
                        "(e.g. `rag-rebuild-index`)."
                    )
                else:
                    mismatches, errors = validate_manifest(
                        manifest=manifest,
                        expected_config=expected_manifest,
                        actual_dimension=int(idx.dim),
                        actual_index_backend=backend,
                    )
                    payload["manifest_errors"] = errors
                    payload["manifest_mismatches"] = [
                        {"key": m.key, "expected": m.expected, "actual": m.actual}
                        for m in mismatches
                    ]
                    if errors:
                        payload["status"] = "corrupt"
                        payload["error"] = "invalid index manifest"
                        payload["hint"] = (
                            "Rebuild the index to repair the manifest (e.g. `rag-rebuild-index`)."
                        )
                    elif mismatches:
                        payload["status"] = "drift"
                        payload["error"] = "index manifest mismatch"
                        payload["hint"] = (
                            "Rebuild the index to remove drift (e.g. `rag-rebuild-index`)."
                        )
        except Exception as e:
            if payload["status"] == "ok":
                payload["status"] = "corrupt"
                payload["error"] = f"manifest read/validation failed: {type(e).__name__}: {e}"
                payload["hint"] = (
                    "Rebuild the index to repair the manifest (e.g. `rag-rebuild-index`)."
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

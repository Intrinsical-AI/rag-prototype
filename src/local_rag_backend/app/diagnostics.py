"""
Operational diagnostics for the app layer (health/readiness/status).

Backward-compatibility:
- `local_rag_backend.diagnostics` re-exports these symbols.
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
    *, index_path: str | Path, id_map_path: str | Path, dim: int | None = None
) -> dict[str, Any]:
    """
    Return best-effort stats for the on-disk retrieval index.

    This is safe to call even when optional deps are missing. It returns:
      - status: ok|missing|corrupt
      - hint: remediation hint when not ok
      - vectors/id_map_len/unique_ids/duplicates/dim/backend when ok (best-effort)
    """
    idx_path = Path(index_path)
    map_path = Path(id_map_path)

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

        return payload
    except Exception as e:
        return {
            "status": "corrupt",
            "index_path": str(idx_path),
            "id_map_path": str(map_path),
            "error": f"{type(e).__name__}: {e}",
            "hint": "Rebuild the index (e.g. `rag-rebuild-index` or POST /api/index/rebuild).",
        }

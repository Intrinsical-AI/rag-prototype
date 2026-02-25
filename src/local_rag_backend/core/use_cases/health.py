"""Application orchestration for health/readiness checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from local_rag_backend.infrastructure.observability.diagnostics import (
    get_document_ids,
    get_documents_count,
    get_history_count,
    get_incomplete_mutation_records_count,
    get_retrieval_index_stats,
)
from local_rag_backend.infrastructure.persistence.sql import base as db_base
from local_rag_backend.infrastructure.persistence.vector.manifest import (
    expected_manifest_config_from_settings,
)

if TYPE_CHECKING:
    from local_rag_backend.settings import Settings


def ping_database() -> None:
    with db_base.engine.connect() as conn:
        conn.execute(text("SELECT 1"))


def check_database(checks: dict[str, Any]) -> bool:
    try:
        ping_database()
        checks["database"] = "ok"
        return True
    except Exception as e:
        checks["database"] = f"failed: {e!s}"
        return False


def check_sql_counts(checks: dict[str, Any]) -> tuple[bool, int | None]:
    docs_count: int | None = None
    is_ready = True
    try:
        docs_count = get_documents_count(db_base.engine)
        checks["documents"] = {"count": docs_count}
    except Exception as e:
        checks["documents"] = f"failed: {e!s}"
        is_ready = False

    try:
        checks["history"] = {"count": get_history_count(db_base.engine)}
    except Exception as e:
        checks["history"] = f"failed: {e!s}"
    return is_ready, docs_count


def _check_retrieval_index_id_set_drift(*, checks: dict[str, Any], settings_obj: Settings) -> bool:
    try:
        db_ids = {str(x) for x in get_document_ids(db_base.engine)}
        index_ids = {
            str(x)
            for x in json.loads(Path(settings_obj.id_map_path).read_text(encoding="utf-8"))
            if str(x).strip()
        }
        stale = sorted(index_ids - db_ids)
        missing = sorted(db_ids - index_ids)
        if not stale and not missing:
            return True
        checks["retrieval_index_drift"] = {
            "stale_in_index": stale[:20],
            "missing_in_index": missing[:20],
            "stale_count": len(stale),
            "missing_count": len(missing),
        }
        checks["retrieval_index"] = (
            "failed: drift detected (ID set mismatch). "
            "Hint: rebuild the index (`rag-rebuild-index` or POST /api/index/rebuild)."
        )
        return False
    except Exception as e:
        checks["retrieval_index_drift"] = f"failed: {e!s}"
        return True


def check_retrieval_index(
    *,
    checks: dict[str, Any],
    docs_count: int | None,
    settings_obj: Settings,
) -> bool:
    if settings_obj.retrieval_mode not in ("dense", "hybrid"):
        return True

    expected_manifest = expected_manifest_config_from_settings(settings_obj)
    stats = get_retrieval_index_stats(
        index_path=settings_obj.index_path,
        id_map_path=settings_obj.id_map_path,
        vector_backend=settings_obj.vector_backend,
        dim=None,
        expected_manifest=expected_manifest,
    )
    checks["retrieval_index_stats"] = stats

    if stats.get("status") != "ok":
        checks["retrieval_index"] = (
            f"failed: {stats.get('status')} "
            f"(index_path={stats.get('index_path')}, id_map_path={stats.get('id_map_path')}). "
            f"Hint: {stats.get('hint')}"
        )
        return False

    checks["retrieval_index"] = "ok"
    if docs_count is None:
        return True

    vectors = int(stats.get("vectors") or 0)
    id_map_len = int(stats.get("id_map_len") or 0)
    if docs_count != id_map_len:
        checks["retrieval_index"] = (
            f"failed: drift detected (documents={docs_count}, vectors={vectors}). "
            "Hint: rebuild the index (`rag-rebuild-index` or POST /api/index/rebuild)."
        )
        return False

    if docs_count <= 5000:
        return _check_retrieval_index_id_set_drift(checks=checks, settings_obj=settings_obj)
    return True


def check_mutation_journal(*, checks: dict[str, Any], settings_obj: Settings) -> None:
    try:
        incomplete = get_incomplete_mutation_records_count(
            coordination_dir=settings_obj.get_coordination_dir()
        )
    except Exception as e:
        checks["mutation_journal"] = {"status": "failed", "error": str(e)}
        return

    if incomplete > 0:
        checks["mutation_journal"] = {"status": "warning", "incomplete_records": int(incomplete)}
        return
    checks["mutation_journal"] = {"status": "ok", "incomplete_records": 0}

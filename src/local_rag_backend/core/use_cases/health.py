"""Application orchestration for health/readiness checks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from local_rag_backend.core.ports import HealthDiagnosticsPort
    from local_rag_backend.settings import Settings

logger = logging.getLogger(__name__)


def _public_retrieval_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """Remove internal paths and raw error text from public readiness data."""
    hidden_keys = {"error", "hint", "id_map_path", "index_path", "manifest_path"}
    public = {key: value for key, value in stats.items() if key not in hidden_keys}
    if "error" in stats:
        public["error"] = "retrieval index unavailable"
    return public


def ping_database(*, diagnostics: HealthDiagnosticsPort) -> None:
    diagnostics.ping_database()


def check_database(*, checks: dict[str, Any], diagnostics: HealthDiagnosticsPort) -> bool:
    try:
        ping_database(diagnostics=diagnostics)
        checks["database"] = "ok"
        return True
    except Exception:
        logger.exception("Database readiness check failed")
        checks["database"] = "failed: unavailable"
        return False


def check_sql_counts(
    *,
    checks: dict[str, Any],
    diagnostics: HealthDiagnosticsPort,
) -> tuple[bool, int | None]:
    docs_count: int | None = None
    is_ready = True
    try:
        docs_count = diagnostics.get_documents_count()
        checks["documents"] = {"count": docs_count}
    except Exception:
        logger.exception("Document-count readiness check failed")
        checks["documents"] = "failed: unavailable"
        is_ready = False

    try:
        checks["history"] = {"count": diagnostics.get_history_count()}
    except Exception:
        logger.exception("History-count readiness check failed")
        checks["history"] = "failed: unavailable"
    return is_ready, docs_count


def _check_retrieval_index_id_set_drift(
    *,
    checks: dict[str, Any],
    settings_obj: Settings,
    diagnostics: HealthDiagnosticsPort,
) -> bool:
    try:
        db_ids = set(diagnostics.get_document_ids())
        index_ids = set(diagnostics.get_index_ids(id_map_path=settings_obj.id_map_path))
        stale = sorted(index_ids - db_ids)
        missing = sorted(db_ids - index_ids)
        if not stale and not missing:
            return True
        checks["retrieval_index_drift"] = {
            "stale_count": len(stale),
            "missing_count": len(missing),
        }
        checks["retrieval_index"] = (
            "failed: drift detected (ID set mismatch). "
            "Hint: rebuild the index (`rag-rebuild-index` or POST /api/index/rebuild)."
        )
        return False
    except Exception:
        logger.exception("Retrieval-index ID drift check failed")
        checks["retrieval_index_drift"] = "failed: unavailable"
        checks["retrieval_index"] = (
            "failed: unable to verify retrieval index drift. "
            "Hint: inspect the index/id_map and rebuild if needed "
            "(`rag-rebuild-index` or POST /api/index/rebuild)."
        )
        return False


def check_retrieval_index(
    *,
    checks: dict[str, Any],
    docs_count: int | None,
    settings_obj: Settings,
    diagnostics: HealthDiagnosticsPort,
    expected_manifest: dict[str, Any] | None = None,
) -> bool:
    if settings_obj.retrieval_mode not in ("dense", "dual", "hybrid"):
        return True
    if str(getattr(settings_obj, "search_backend", "local_split")) != "local_split":
        return True

    stats = diagnostics.get_retrieval_index_stats(
        index_path=settings_obj.index_path,
        id_map_path=settings_obj.id_map_path,
        vector_backend=settings_obj.vector_backend,
        dim=None,
        expected_manifest=expected_manifest,
    )
    checks["retrieval_index_stats"] = _public_retrieval_stats(stats)

    if stats.get("status") != "ok":
        checks["retrieval_index"] = (
            f"failed: {stats.get('status')} "
            "Hint: inspect the retrieval index and rebuild it if needed "
            "(`rag-rebuild-index` or POST /api/index/rebuild)."
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
        return _check_retrieval_index_id_set_drift(
            checks=checks,
            settings_obj=settings_obj,
            diagnostics=diagnostics,
        )
    return True


def check_mutation_journal(
    *,
    checks: dict[str, Any],
    settings_obj: Settings,
    diagnostics: HealthDiagnosticsPort,
) -> None:
    if str(getattr(settings_obj, "persistence_backend", "local_split")) == "elasticsearch":
        checks["mutation_journal"] = {"status": "not_applicable"}
        return

    try:
        incomplete = diagnostics.get_incomplete_mutation_records_count(
            coordination_dir=settings_obj.get_coordination_dir()
        )
    except Exception:
        logger.exception("Mutation-journal readiness check failed")
        checks["mutation_journal"] = {"status": "failed"}
        return

    if incomplete > 0:
        checks["mutation_journal"] = {"status": "warning", "incomplete_records": int(incomplete)}
        return
    checks["mutation_journal"] = {"status": "ok", "incomplete_records": 0}

"""
Bounded router for health and readiness endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from local_rag_backend.app.blocking import run_blocking
from local_rag_backend.app.dependencies import get_rag_service
from local_rag_backend.app.diagnostics import (
    get_document_ids,
    get_documents_count,
    get_history_count,
    get_retrieval_index_stats,
)
from local_rag_backend.app.wiring import get_available_llm_providers
from local_rag_backend.infrastructure.persistence.faiss.manifest import (
    expected_manifest_config_from_settings,
)
from local_rag_backend.infrastructure.persistence.sqlalchemy import base as db_base
from local_rag_backend.settings import settings

router = APIRouter()


@router.get("/health", tags=["Health"], summary="Health check endpoint")
async def health_check() -> dict[str, str]:
    """Basic health check for service availability (e.g., Docker/K8s)."""
    try:
        with db_base.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection failed: {e!s}")


@router.get("/ready", tags=["Health"], summary="Readiness check endpoint")
async def readiness_check() -> dict[str, Any]:
    """Check if all dependencies are ready to handle requests."""
    checks: dict[str, Any] = {}
    is_ready = True
    docs_count: int | None = None

    try:
        with db_base.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"failed: {e!s}"
        is_ready = False

    if checks.get("database") == "ok":
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

    try:
        service = await get_rag_service()
        checks["rag_service"] = "ok" if service else "failed: not initialized"
        if not service:
            is_ready = False
    except Exception as e:
        checks["rag_service"] = f"failed: {e!s}"
        is_ready = False

    llm_providers = get_available_llm_providers()
    if not llm_providers:
        checks["llm_providers"] = "failed: no providers configured"
        is_ready = False
    else:
        checks["llm_providers"] = llm_providers

    if settings.retrieval_mode in ["dense", "hybrid"]:
        expected_manifest = expected_manifest_config_from_settings(settings)
        stats = get_retrieval_index_stats(
            index_path=settings.index_path,
            id_map_path=settings.id_map_path,
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
            is_ready = False
        else:
            checks["retrieval_index"] = "ok"
            if docs_count is not None:
                vectors = int(stats.get("vectors") or 0)
                id_map_len = int(stats.get("id_map_len") or 0)
                if docs_count != id_map_len:
                    checks["retrieval_index"] = (
                        f"failed: drift detected (documents={docs_count}, vectors={vectors}). "
                        "Hint: rebuild the index (`rag-rebuild-index` or POST /api/index/rebuild)."
                    )
                    is_ready = False
                else:
                    if docs_count <= 5000:
                        try:
                            db_ids = set(get_document_ids(db_base.engine))
                            index_ids = set(
                                json.loads(Path(settings.id_map_path).read_text(encoding="utf-8"))
                            )
                            stale = sorted(index_ids - db_ids)
                            missing = sorted(db_ids - index_ids)
                            if stale or missing:
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
                                is_ready = False
                        except Exception as e:
                            checks["retrieval_index_drift"] = f"failed: {e!s}"

    response_payload = {"status": "ready" if is_ready else "not_ready", "checks": checks}
    if not is_ready:
        raise HTTPException(status_code=503, detail=response_payload)
    return response_payload


@router.get("/health/ollama", tags=["Health"], summary="Ollama server health check")
async def ollama_health_check() -> dict[str, Any]:
    """Checks if the Ollama server is running and accessible."""
    try:
        response = await run_blocking(
            requests.get, settings.ollama_base_url, timeout=5, task_type="network"
        )
        response.raise_for_status()
        return {"status": "ok", "url": settings.ollama_base_url}
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama server not accessible at {settings.ollama_base_url}. Error: {e}",
        )

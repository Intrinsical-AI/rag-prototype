"""
Bounded router for health and readiness endpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from local_rag_backend.composition.adapters import (
    get_available_llm_providers as get_available_llm_providers_from_settings,
)
from local_rag_backend.core.use_cases.errors import ServiceUnavailableError
from local_rag_backend.core.use_cases.health import (
    check_database,
    check_mutation_journal,
    check_retrieval_index,
    check_sql_counts,
    ping_database,
)
from local_rag_backend.http.dependencies import (
    get_app_container_dependency,
    get_rag_service,
    get_settings_dependency,
)
from local_rag_backend.infrastructure.concurrency.blocking import run_blocking

if TYPE_CHECKING:
    from local_rag_backend.composition.container import AppContainer
    from local_rag_backend.settings import Settings

router = APIRouter()


async def _check_rag_service(checks: dict[str, Any]) -> bool:
    try:
        service = await get_rag_service()
        checks["rag_service"] = "ok" if service else "failed: not initialized"
        return bool(service)
    except Exception as e:
        checks["rag_service"] = f"failed: {e!s}"
        return False


def _check_llm_providers(*, checks: dict[str, Any], settings_obj: Settings) -> bool:
    llm_providers = get_available_llm_providers_from_settings(settings_obj=settings_obj)
    if not llm_providers:
        checks["llm_providers"] = "failed: no providers configured"
        return False
    checks["llm_providers"] = llm_providers
    return True


@router.get("/healthz", tags=["Health"], summary="Health probe endpoint")
async def health_check(
    container: AppContainer = Depends(get_app_container_dependency),
) -> dict[str, str]:
    """Basic health check for service availability (e.g., Docker/K8s)."""
    try:
        readiness_bundle = container.build_health_readiness_bundle()
        ping_database(diagnostics=readiness_bundle.diagnostics)
        return {"status": "healthy"}
    except Exception as e:
        raise ServiceUnavailableError(f"Database connection failed: {e!s}") from e


@router.get("/readyz", tags=["Health"], summary="Readiness probe endpoint")
async def readiness_check(
    settings_obj: Settings = Depends(get_settings_dependency),
    container: AppContainer = Depends(get_app_container_dependency),
) -> Any:
    """Check if all dependencies are ready to handle requests."""
    checks: dict[str, Any] = {}
    is_ready = True
    docs_count: int | None = None
    readiness_bundle = container.build_health_readiness_bundle()
    diagnostics = readiness_bundle.diagnostics
    expected_manifest = readiness_bundle.expected_manifest

    db_ok = check_database(checks=checks, diagnostics=diagnostics)
    if not db_ok:
        is_ready = False
    if db_ok:
        db_ready, docs_count = check_sql_counts(checks=checks, diagnostics=diagnostics)
        if not db_ready:
            is_ready = False

    if not await _check_rag_service(checks):
        is_ready = False
    if not _check_llm_providers(checks=checks, settings_obj=settings_obj):
        is_ready = False
    if not check_retrieval_index(
        checks=checks,
        docs_count=docs_count,
        settings_obj=settings_obj,
        diagnostics=diagnostics,
        expected_manifest=expected_manifest,
    ):
        is_ready = False
    check_mutation_journal(checks=checks, settings_obj=settings_obj, diagnostics=diagnostics)

    response_payload = {"status": "ready" if is_ready else "not_ready", "checks": checks}
    if not is_ready:
        return JSONResponse(status_code=503, content=response_payload)
    return response_payload


@router.get("/healthz/ollama", tags=["Health"], summary="Ollama server health check")
async def ollama_health_check(
    settings_obj: Settings = Depends(get_settings_dependency),
) -> dict[str, Any]:
    """Checks if the Ollama server is running and accessible."""
    try:
        response = await run_blocking(
            httpx.get, settings_obj.ollama_base_url, timeout=5, task_type="network"
        )
        response.raise_for_status()
        return {"status": "ok", "url": settings_obj.ollama_base_url}
    except httpx.HTTPError as e:
        raise ServiceUnavailableError(
            f"Ollama server not accessible at {settings_obj.ollama_base_url}. Error: {e}",
        )

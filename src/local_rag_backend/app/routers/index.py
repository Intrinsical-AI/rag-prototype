"""
Bounded router for index maintenance endpoints.
"""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, HTTPException

from local_rag_backend.app.blocking import run_blocking
from local_rag_backend.app.dependencies import reset_rag_service
from local_rag_backend.app.schemas import RebuildIndexResponse
from local_rag_backend.app.services import index as index_service
from local_rag_backend.app.wiring import index_mutation_ports, run_multi_store_write_locked
from local_rag_backend.settings import settings

router = APIRouter()


@router.post("/index/rebuild", response_model=RebuildIndexResponse)
async def rebuild_index() -> RebuildIndexResponse:
    if settings.retrieval_mode not in ("dense", "hybrid"):
        raise HTTPException(status_code=400, detail="Index rebuild requires dense or hybrid mode.")

    def _rebuild_operation() -> int:
        return index_service.rebuild_index_sync(
            settings_obj=settings,
            ports=index_mutation_ports(),
        )

    try:
        indexed = cast(
            "int",
            await run_blocking(
                run_multi_store_write_locked,
                _rebuild_operation,
                task_type="mutation",
            ),
        )
        return RebuildIndexResponse(indexed=indexed)
    finally:
        reset_rag_service()

"""
Bounded router for index maintenance endpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fastapi import APIRouter, Depends

from local_rag_backend.app.application import index as index_service
from local_rag_backend.app.application.mutations import run_api_mutation
from local_rag_backend.app.blocking import run_blocking
from local_rag_backend.app.dependencies import (
    get_app_container_dependency,
    get_settings_dependency,
    reset_rag_service,
)
from local_rag_backend.app.errors import BadRequestError
from local_rag_backend.app.schemas.index import RebuildIndexResponse

if TYPE_CHECKING:
    from local_rag_backend.app.container import AppContainer
    from local_rag_backend.settings import Settings

router = APIRouter()


@router.post("/index/rebuild", response_model=RebuildIndexResponse)
async def rebuild_index(
    settings_obj: Settings = Depends(get_settings_dependency),
    container: AppContainer = Depends(get_app_container_dependency),
) -> RebuildIndexResponse:
    if settings_obj.retrieval_mode not in ("dense", "hybrid"):
        raise BadRequestError("Index rebuild requires dense or hybrid mode.")

    def _rebuild_operation() -> int:
        return index_service.rebuild_index_sync(
            settings_obj=settings_obj,
            ports=container.index_mutation_ports(),
        )

    indexed = cast(
        "int",
        await run_api_mutation(
            operation=_rebuild_operation,
            run_locked=container.run_multi_store_write_locked,
            reset_after=reset_rag_service,
            run_blocking_fn=run_blocking,
        ),
    )
    return RebuildIndexResponse(indexed=indexed)

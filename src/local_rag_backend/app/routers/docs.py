"""
Bounded router for documents mutation/query endpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, cast

from fastapi import APIRouter, Body, Depends, File, Query, UploadFile

from local_rag_backend.app.application.docs import (
    ImportDocsOutcome,
    ImportFileTooLargeError,
    ImportPayloadEmptyError,
    InvalidImportPayloadError,
    UnsupportedImportFormatError,
    execute_import_docs_sync,
    list_docs_page_sync,
)
from local_rag_backend.app.application.mutations import run_api_mutation
from local_rag_backend.app.blocking import run_blocking
from local_rag_backend.app.composition import DEFAULT_DENSE_BACKEND_MESSAGE
from local_rag_backend.app.dependencies import (
    get_app_container_dependency,
    get_db,
    get_settings_dependency,
    reset_rag_service,
)
from local_rag_backend.app.errors import (
    BadRequestError,
    ConflictError,
    PayloadTooLargeError,
    UnprocessableEntityError,
)
from local_rag_backend.app.observability import Timer, log_event, observe_ingest
from local_rag_backend.app.schemas.docs import (
    DeleteDocsByExternalIdRequest,
    DeleteDocsByExternalIdResponse,
    DeleteDocsRequest,
    DeleteDocsResponse,
    ImportResponse,
    IngestRequest,
    IngestResponse,
    UpsertDocResult,
    UpsertDocsRequest,
    UpsertDocsResponse,
)
from local_rag_backend.app.schemas.shared import DocumentInDB
from local_rag_backend.app.services import docs as docs_service
from local_rag_backend.core.errors import EmbeddingsBackendUnavailableError

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from local_rag_backend.app.container import AppContainer
    from local_rag_backend.app.services.results import (
        DeleteDocsByExternalIdSummary,
        DeleteDocsSummary,
        UpsertDocsSummary,
    )
    from local_rag_backend.settings import Settings

router = APIRouter()


@router.get("/docs", response_model=list[DocumentInDB])
async def list_docs(
    limit: int = Query(100, ge=1, le=1000, description="Max number of docs"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
) -> list[DocumentInDB]:
    docs = list_docs_page_sync(db=db, limit=limit, offset=offset)
    return [DocumentInDB.model_validate(item) for item in docs]


def _map_ingest_error(exc: Exception) -> BadRequestError | None:
    if isinstance(exc, EmbeddingsBackendUnavailableError):
        return BadRequestError(DEFAULT_DENSE_BACKEND_MESSAGE)
    return None


def _map_upsert_error(exc: Exception) -> ConflictError | BadRequestError | None:
    if isinstance(exc, docs_service.TombstonedExternalIdsError):
        return ConflictError(str(exc))
    if isinstance(exc, ValueError):
        return BadRequestError(str(exc))
    return None


def _map_import_error(
    exc: Exception,
) -> PayloadTooLargeError | UnprocessableEntityError | BadRequestError | None:
    if isinstance(exc, ImportFileTooLargeError):
        return PayloadTooLargeError(str(exc))
    if isinstance(
        exc, ImportPayloadEmptyError | UnsupportedImportFormatError | InvalidImportPayloadError
    ):
        return UnprocessableEntityError(str(exc))
    if isinstance(exc, EmbeddingsBackendUnavailableError):
        return BadRequestError(DEFAULT_DENSE_BACKEND_MESSAGE)
    return None


@router.post("/docs", response_model=IngestResponse)
async def ingest_docs(
    payload: Annotated[IngestRequest, Body(...)],
    container: AppContainer = Depends(get_app_container_dependency),
    settings_obj: Settings = Depends(get_settings_dependency),
) -> IngestResponse:
    texts = [t.strip() for t in payload.texts if t and t.strip()]
    if not texts:
        return IngestResponse(count=0, ids=[])
    t = Timer()

    def _ingest_operation() -> list[int]:
        return docs_service.ingest_docs_sync(
            texts=texts,
            settings_obj=settings_obj,
            ports=container.docs_mutation_ports(
                missing_backend_message=DEFAULT_DENSE_BACKEND_MESSAGE
            ),
        )

    ok = False
    ids: list[int] = []
    try:
        ids = cast(
            "list[int]",
            await run_api_mutation(
                operation=_ingest_operation,
                run_locked=container.run_multi_store_write_locked,
                reset_after=reset_rag_service,
                run_blocking_fn=run_blocking,
                map_error=_map_ingest_error,
            ),
        )
        ok = True
        return IngestResponse(count=len(ids), ids=ids)
    finally:
        observe_ingest(source="api:/docs", ok=ok, inserted=len(ids))
        log_event(
            "rag_ingest",
            source="api:/docs",
            ok=ok,
            input_texts=len(texts),
            inserted=len(ids),
            retrieval_mode=str(settings_obj.retrieval_mode),
            reranker_enabled=bool(settings_obj.enable_reranker),
            duration_ms=int(1000 * t.seconds()),
        )


@router.post("/docs/delete_by_external_id", response_model=DeleteDocsByExternalIdResponse)
async def delete_docs_by_external_id(
    payload: Annotated[DeleteDocsByExternalIdRequest, Body(...)],
    container: AppContainer = Depends(get_app_container_dependency),
    settings_obj: Settings = Depends(get_settings_dependency),
) -> DeleteDocsByExternalIdResponse:
    def _delete_operation() -> DeleteDocsByExternalIdSummary:
        return docs_service.delete_docs_by_external_id_sync(
            external_ids=payload.external_ids,
            settings_obj=settings_obj,
            ports=container.docs_mutation_ports(
                missing_backend_message=DEFAULT_DENSE_BACKEND_MESSAGE
            ),
        )

    summary = cast(
        "DeleteDocsByExternalIdSummary",
        await run_api_mutation(
            operation=_delete_operation,
            run_locked=container.run_multi_store_write_locked,
            reset_after=reset_rag_service,
            run_blocking_fn=run_blocking,
        ),
    )
    return DeleteDocsByExternalIdResponse(
        deleted_sql=summary.deleted_sql,
        deleted_index=summary.deleted_index,
        tombstoned=summary.tombstoned,
        missing_external_ids=summary.missing_external_ids,
        rebuilt_index=summary.rebuilt_index,
    )


@router.post("/docs/delete", response_model=DeleteDocsResponse)
async def delete_docs(
    payload: Annotated[DeleteDocsRequest, Body(...)],
    container: AppContainer = Depends(get_app_container_dependency),
    settings_obj: Settings = Depends(get_settings_dependency),
) -> DeleteDocsResponse:
    def _delete_operation() -> DeleteDocsSummary:
        return docs_service.delete_docs_sync(
            ids=payload.ids,
            settings_obj=settings_obj,
            ports=container.docs_mutation_ports(
                missing_backend_message=DEFAULT_DENSE_BACKEND_MESSAGE
            ),
        )

    summary = cast(
        "DeleteDocsSummary",
        await run_api_mutation(
            operation=_delete_operation,
            run_locked=container.run_multi_store_write_locked,
            reset_after=reset_rag_service,
            run_blocking_fn=run_blocking,
        ),
    )
    return DeleteDocsResponse(
        deleted_sql=summary.deleted_sql,
        deleted_index=summary.deleted_index,
        rebuilt_index=summary.rebuilt_index,
    )


@router.post("/docs/upsert", response_model=UpsertDocsResponse)
async def upsert_docs(
    payload: Annotated[UpsertDocsRequest, Body(...)],
    container: AppContainer = Depends(get_app_container_dependency),
    settings_obj: Settings = Depends(get_settings_dependency),
) -> UpsertDocsResponse:
    def _upsert_operation() -> UpsertDocsSummary:
        return docs_service.upsert_docs_sync(
            docs=payload.docs,
            settings_obj=settings_obj,
            ports=container.docs_mutation_ports(
                missing_backend_message=DEFAULT_DENSE_BACKEND_MESSAGE
            ),
        )

    summary = cast(
        "UpsertDocsSummary",
        await run_api_mutation(
            operation=_upsert_operation,
            run_locked=container.run_multi_store_write_locked,
            reset_after=reset_rag_service,
            run_blocking_fn=run_blocking,
            map_error=_map_upsert_error,
        ),
    )
    return UpsertDocsResponse(
        inserted=summary.inserted,
        updated=summary.updated,
        unchanged=summary.unchanged,
        rebuilt_index=summary.rebuilt_index,
        results=[
            UpsertDocResult(
                external_id=r.external_id,
                id=r.id,
                action=r.action,
                content_changed=r.content_changed,
            )
            for r in summary.results
        ],
    )


@router.post("/docs/import", response_model=ImportResponse)
async def import_docs(
    file: UploadFile = File(..., description="JSON file exported from ChatGPT or Google Gemini"),
    container: AppContainer = Depends(get_app_container_dependency),
    settings_obj: Settings = Depends(get_settings_dependency),
) -> ImportResponse:
    """
    Import conversations from a ChatGPT or Gemini export JSON file.

    Accepts multipart/form-data with a single file field named 'file'.
    Detects the export format automatically and ingests each message as a separate document.
    """
    raw = await file.read()
    t = Timer()
    ok = False
    outcome: ImportDocsOutcome | None = None
    try:

        def _import_operation() -> ImportDocsOutcome:
            return execute_import_docs_sync(
                raw=raw,
                settings_obj=settings_obj,
                ports=container.docs_mutation_ports(
                    missing_backend_message=DEFAULT_DENSE_BACKEND_MESSAGE
                ),
            )

        outcome = cast(
            "ImportDocsOutcome",
            await run_api_mutation(
                operation=_import_operation,
                run_locked=container.run_multi_store_write_locked,
                reset_after=reset_rag_service,
                run_blocking_fn=run_blocking,
                map_error=_map_import_error,
            ),
        )
        ok = True
        return ImportResponse(
            count=outcome.count,
            ids=outcome.ids,
            format_detected=outcome.format_detected,
        )
    finally:
        format_detected = outcome.format_detected if outcome is not None else "unknown"
        inserted = outcome.count if outcome is not None else 0
        input_texts = outcome.input_texts if outcome is not None else 0
        observe_ingest(source=f"api:/docs/import:{format_detected}", ok=ok, inserted=inserted)
        log_event(
            "rag_ingest",
            source="api:/docs/import",
            format_detected=format_detected,
            ok=ok,
            input_texts=input_texts,
            inserted=inserted,
            retrieval_mode=str(settings_obj.retrieval_mode),
            reranker_enabled=bool(settings_obj.enable_reranker),
            duration_ms=int(1000 * t.seconds()),
        )

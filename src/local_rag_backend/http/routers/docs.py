"""
Bounded router for documents mutation/query endpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, cast

from fastapi import APIRouter, Body, Depends, File, Query, UploadFile

from local_rag_backend.composition.adapters import DEFAULT_DENSE_BACKEND_MESSAGE
from local_rag_backend.core.errors import EmbeddingsBackendUnavailableError
from local_rag_backend.core.use_cases.docs_import import (
    ImportDocsOutcome,
    ImportFileTooLargeError,
    ImportPayloadEmptyError,
    InvalidImportPayloadError,
    UnsupportedImportFormatError,
    execute_import_docs_sync,
)
from local_rag_backend.core.use_cases.docs_ingest import ingest_docs_sync
from local_rag_backend.core.use_cases.docs_mutation import (
    MutationCoordinator,
    MutationIntent,
    MutationUpsertInput,
)
from local_rag_backend.core.use_cases.docs_query import list_docs_page_sync
from local_rag_backend.core.use_cases.errors import (
    BadRequestError,
    PayloadTooLargeError,
    UnprocessableEntityError,
)
from local_rag_backend.core.use_cases.mutations import run_api_mutation
from local_rag_backend.http.dependencies import (
    get_app_container_dependency,
    get_db,
    get_settings_dependency,
    reset_rag_service,
)
from local_rag_backend.http.schemas.docs import (
    DocsMutateRequest,
    DocsMutateResponse,
    ImportResponse,
    IngestRequest,
    IngestResponse,
    UpsertDocResult,
)
from local_rag_backend.http.schemas.shared import DocumentInDB
from local_rag_backend.infrastructure.concurrency.blocking import run_blocking
from local_rag_backend.infrastructure.observability.observability import (
    Timer,
    log_event,
    observe_ingest,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from local_rag_backend.composition.container import AppContainer
    from local_rag_backend.core.use_cases.results import MutationSummary
    from local_rag_backend.settings import Settings

router = APIRouter()


@router.get("/docs", response_model=list[DocumentInDB])
async def list_docs(
    limit: int = Query(100, ge=1, le=1000, description="Max number of docs"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
    container: AppContainer = Depends(get_app_container_dependency),
) -> list[DocumentInDB]:
    query_bundle = container.build_docs_query_bundle(db=db)
    docs = list_docs_page_sync(
        docs_reader=query_bundle.docs_reader,
        limit=limit,
        offset=offset,
    )
    return [DocumentInDB(id=item.id, content=item.content) for item in docs]


def _map_ingest_error(exc: Exception) -> BadRequestError | None:
    if isinstance(exc, EmbeddingsBackendUnavailableError):
        return BadRequestError(DEFAULT_DENSE_BACKEND_MESSAGE)
    return None


def _map_mutation_error(exc: Exception) -> BadRequestError | None:
    if isinstance(exc, ValueError):
        return BadRequestError(str(exc))
    if isinstance(exc, EmbeddingsBackendUnavailableError):
        return BadRequestError(DEFAULT_DENSE_BACKEND_MESSAGE)
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


@router.post("/docs/mutate", response_model=DocsMutateResponse)
async def mutate_docs(
    payload: Annotated[DocsMutateRequest, Body(...)],
    container: AppContainer = Depends(get_app_container_dependency),
    settings_obj: Settings = Depends(get_settings_dependency),
) -> DocsMutateResponse:
    mutation_bundle = container.build_docs_mutation_bundle(
        missing_backend_message=DEFAULT_DENSE_BACKEND_MESSAGE
    )
    execution_bundle = container.build_mutation_execution_bundle(run_blocking_fn=run_blocking)

    def _mutate_operation() -> MutationSummary:
        coordinator = MutationCoordinator(settings_obj=settings_obj, ports=mutation_bundle.ports)
        return coordinator.execute(
            MutationIntent(
                op_id=str(payload.op_id or "").strip(),
                upserts=tuple(
                    MutationUpsertInput(
                        external_id=item.external_id,
                        content=item.content,
                        source_id=item.source_id,
                        metadata=item.metadata,
                    )
                    for item in payload.upserts
                ),
                delete_ids=tuple(payload.delete_ids),
                delete_external_ids=tuple(payload.delete_external_ids),
                source="api:/docs/mutate",
            )
        )

    summary = cast(
        "MutationSummary",
        await run_api_mutation(
            operation=_mutate_operation,
            run_locked=execution_bundle.run_locked,
            reset_after=reset_rag_service,
            blocking_executor=execution_bundle.blocking_executor,
            map_error=_map_mutation_error,
        ),
    )
    return DocsMutateResponse(
        op_id=summary.op_id,
        inserted=summary.inserted,
        updated=summary.updated,
        unchanged=summary.unchanged,
        deleted_sql=summary.deleted_sql,
        deleted_index=summary.deleted_index,
        tombstoned=summary.tombstoned,
        missing_external_ids=list(summary.missing_external_ids or []),
        index_rebuilt=summary.index_rebuilt,
        index_doc_count=summary.index_doc_count,
        results=[
            UpsertDocResult(
                external_id=r.external_id,
                id=r.id,
                action=r.action,
                content_changed=r.content_changed,
            )
            for r in list(summary.results or [])
        ],
    )


@router.post("/docs", response_model=IngestResponse)
async def ingest_docs(
    payload: Annotated[IngestRequest, Body(...)],
    container: AppContainer = Depends(get_app_container_dependency),
    settings_obj: Settings = Depends(get_settings_dependency),
) -> IngestResponse:
    mutation_bundle = container.build_docs_mutation_bundle(
        missing_backend_message=DEFAULT_DENSE_BACKEND_MESSAGE
    )
    execution_bundle = container.build_mutation_execution_bundle(run_blocking_fn=run_blocking)
    texts = [t.strip() for t in payload.texts if t and t.strip()]
    if not texts:
        return IngestResponse(count=0, ids=[])
    t = Timer()

    def _ingest_operation() -> list[str]:
        return ingest_docs_sync(
            texts=texts,
            settings_obj=settings_obj,
            ports=mutation_bundle.ports,
            source="api:/docs",
        )

    ok = False
    ids: list[str] = []
    try:
        ids = cast(
            "list[str]",
            await run_api_mutation(
                operation=_ingest_operation,
                run_locked=execution_bundle.run_locked,
                reset_after=reset_rag_service,
                blocking_executor=execution_bundle.blocking_executor,
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
    mutation_bundle = container.build_docs_mutation_bundle(
        missing_backend_message=DEFAULT_DENSE_BACKEND_MESSAGE
    )
    execution_bundle = container.build_mutation_execution_bundle(run_blocking_fn=run_blocking)
    t = Timer()
    ok = False
    outcome: ImportDocsOutcome | None = None
    try:

        def _import_operation() -> ImportDocsOutcome:
            return execute_import_docs_sync(
                raw=raw,
                settings_obj=settings_obj,
                ports=mutation_bundle.ports,
                import_loader=mutation_bundle.import_loader,
            )

        outcome = cast(
            "ImportDocsOutcome",
            await run_api_mutation(
                operation=_import_operation,
                run_locked=execution_bundle.run_locked,
                reset_after=reset_rag_service,
                blocking_executor=execution_bundle.blocking_executor,
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

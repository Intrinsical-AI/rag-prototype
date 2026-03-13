"""
Bounded router for documents mutation/query endpoints.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, cast

from fastapi import APIRouter, Body, Depends, File, Query, UploadFile

from local_rag_backend.composition.adapters import DEFAULT_DENSE_BACKEND_MESSAGE
from local_rag_backend.core.errors import EmbeddingsBackendUnavailableError
from local_rag_backend.core.use_cases.docs_import import (
    DEFAULT_IMPORT_MAX_BYTES,
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
from local_rag_backend.core.use_cases.errors import (
    BadRequestError,
    PayloadTooLargeError,
    UnprocessableEntityError,
)
from local_rag_backend.core.use_cases.mutations import run_api_mutation
from local_rag_backend.http.dependencies import (
    get_app_container_dependency,
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
    from local_rag_backend.composition.container import AppContainer
    from local_rag_backend.core.use_cases.results import MutationSummary
    from local_rag_backend.settings import Settings

router = APIRouter()


def _map_docs_error(
    exc: Exception,
    *,
    operation: str,
) -> PayloadTooLargeError | UnprocessableEntityError | BadRequestError | None:
    if isinstance(exc, EmbeddingsBackendUnavailableError):
        return BadRequestError(DEFAULT_DENSE_BACKEND_MESSAGE)
    if operation == "mutate" and isinstance(exc, ValueError):
        return BadRequestError(str(exc))
    if operation == "import":
        if isinstance(exc, ImportFileTooLargeError):
            return PayloadTooLargeError(str(exc))
        if isinstance(
            exc, ImportPayloadEmptyError | UnsupportedImportFormatError | InvalidImportPayloadError
        ):
            return UnprocessableEntityError(str(exc))
    return None


async def _run_docs_mutation_operation(
    *,
    operation: Callable[[], Any],
    container: AppContainer,
    map_error: Callable[
        [Exception], PayloadTooLargeError | UnprocessableEntityError | BadRequestError | None
    ],
) -> Any:
    return await run_api_mutation(
        operation=operation,
        run_locked=lambda fn: fn(),
        reset_after=reset_rag_service,
        blocking_executor=container.blocking_executor(run_blocking_fn=run_blocking),
        map_error=map_error,
    )


@router.get("/docs", response_model=list[DocumentInDB])
async def list_docs(
    limit: int = Query(100, ge=1, le=1000, description="Max number of docs"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    container: AppContainer = Depends(get_app_container_dependency),
) -> list[DocumentInDB]:
    docs_reader = container.build_docs_read_port()
    docs = docs_reader.list_docs_page(limit=limit, offset=offset)
    return [DocumentInDB(id=item.id, content=item.content) for item in docs]


async def _read_upload_with_limit(
    *,
    file: UploadFile,
    max_bytes: int = DEFAULT_IMPORT_MAX_BYTES,
    chunk_bytes: int = 1024 * 1024,
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    max_allowed = int(max_bytes)
    read_size = max(1, int(chunk_bytes))

    while True:
        chunk = await file.read(read_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_allowed:
            raise ImportFileTooLargeError(max_bytes=max_allowed)
        chunks.append(bytes(chunk))
    return b"".join(chunks)


@router.post("/docs/mutate", response_model=DocsMutateResponse)
async def mutate_docs(
    payload: Annotated[DocsMutateRequest, Body(...)],
    container: AppContainer = Depends(get_app_container_dependency),
    settings_obj: Settings = Depends(get_settings_dependency),
) -> DocsMutateResponse:
    mutation_bundle = container.build_docs_mutation_bundle(
        missing_backend_message=DEFAULT_DENSE_BACKEND_MESSAGE
    )

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
        await _run_docs_mutation_operation(
            operation=_mutate_operation,
            container=container,
            map_error=lambda exc: _map_docs_error(exc, operation="mutate"),
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
            await _run_docs_mutation_operation(
                operation=_ingest_operation,
                container=container,
                map_error=lambda exc: _map_docs_error(exc, operation="ingest"),
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
    raw = await _read_upload_with_limit(file=file)
    mutation_bundle = container.build_docs_mutation_bundle(
        missing_backend_message=DEFAULT_DENSE_BACKEND_MESSAGE
    )
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
            await _run_docs_mutation_operation(
                operation=_import_operation,
                container=container,
                map_error=lambda exc: _map_docs_error(exc, operation="import"),
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

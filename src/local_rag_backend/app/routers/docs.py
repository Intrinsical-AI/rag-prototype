"""
Bounded router for documents mutation/query endpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, cast

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from local_rag_backend.app.blocking import run_blocking
from local_rag_backend.app.dependencies import reset_rag_service
from local_rag_backend.app.observability import Timer, log_event, observe_ingest
from local_rag_backend.app.schemas import (
    DeleteDocsByExternalIdRequest,
    DeleteDocsByExternalIdResponse,
    DeleteDocsRequest,
    DeleteDocsResponse,
    DocumentInDB,
    IngestRequest,
    IngestResponse,
    UpsertDocResult,
    UpsertDocsRequest,
    UpsertDocsResponse,
)
from local_rag_backend.app.services import docs as docs_service
from local_rag_backend.app.wiring import (
    DENSE_BACKEND_ERROR_MESSAGE,
    docs_mutation_ports,
    run_multi_store_write_locked,
)
from local_rag_backend.core.errors import EmbeddingsBackendUnavailableError
from local_rag_backend.infrastructure.persistence.sqlalchemy.base import get_db
from local_rag_backend.infrastructure.persistence.sqlalchemy.models import Document as DbDocument
from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

router = APIRouter()


@router.get("/docs", response_model=list[DocumentInDB])
async def list_docs(
    limit: int = Query(100, ge=1, le=1000, description="Max number of docs"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
) -> list[DocumentInDB]:
    docs = db.query(DbDocument).order_by(DbDocument.id.asc()).offset(offset).limit(limit).all()
    return [DocumentInDB.model_validate(d) for d in docs]


@router.post("/docs", response_model=IngestResponse)
async def ingest_docs(payload: Annotated[IngestRequest, Body(...)]) -> IngestResponse:
    texts = [t.strip() for t in payload.texts if t and t.strip()]
    if not texts:
        return IngestResponse(count=0, ids=[])
    t = Timer()

    def _ingest_operation() -> list[int]:
        return docs_service.ingest_docs_sync(
            texts=texts,
            settings_obj=settings,
            ports=docs_mutation_ports(),
        )

    ok = False
    ids: list[int] = []
    try:
        ids = cast(
            "list[int]",
            await run_blocking(
                run_multi_store_write_locked,
                _ingest_operation,
                task_type="mutation",
            ),
        )
        ok = True
        return IngestResponse(count=len(ids), ids=ids)
    except EmbeddingsBackendUnavailableError as e:
        raise HTTPException(status_code=400, detail=DENSE_BACKEND_ERROR_MESSAGE) from e
    finally:
        observe_ingest(source="api:/docs", ok=ok, inserted=len(ids))
        log_event(
            "rag_ingest",
            source="api:/docs",
            ok=ok,
            input_texts=len(texts),
            inserted=len(ids),
            retrieval_mode=str(settings.retrieval_mode),
            reranker_enabled=bool(settings.enable_reranker),
            duration_ms=int(1000 * t.seconds()),
        )
        reset_rag_service()


@router.post("/docs/delete_by_external_id", response_model=DeleteDocsByExternalIdResponse)
async def delete_docs_by_external_id(
    payload: Annotated[DeleteDocsByExternalIdRequest, Body(...)],
) -> DeleteDocsByExternalIdResponse:
    def _delete_operation() -> docs_service.DeleteDocsByExternalIdSummary:
        return docs_service.delete_docs_by_external_id_sync(
            external_ids=payload.external_ids,
            settings_obj=settings,
            ports=docs_mutation_ports(),
        )

    try:
        summary = cast(
            "docs_service.DeleteDocsByExternalIdSummary",
            await run_blocking(
                run_multi_store_write_locked,
                _delete_operation,
                task_type="mutation",
            ),
        )
        return DeleteDocsByExternalIdResponse(
            deleted_sql=summary.deleted_sql,
            deleted_index=summary.deleted_index,
            tombstoned=summary.tombstoned,
            missing_external_ids=summary.missing_external_ids,
            rebuilt_index=summary.rebuilt_index,
        )
    finally:
        reset_rag_service()


@router.post("/docs/delete", response_model=DeleteDocsResponse)
async def delete_docs(payload: Annotated[DeleteDocsRequest, Body(...)]) -> DeleteDocsResponse:
    def _delete_operation() -> docs_service.DeleteDocsSummary:
        return docs_service.delete_docs_sync(
            ids=payload.ids,
            settings_obj=settings,
            ports=docs_mutation_ports(),
        )

    try:
        summary = cast(
            "docs_service.DeleteDocsSummary",
            await run_blocking(
                run_multi_store_write_locked,
                _delete_operation,
                task_type="mutation",
            ),
        )
        return DeleteDocsResponse(
            deleted_sql=summary.deleted_sql,
            deleted_index=summary.deleted_index,
            rebuilt_index=summary.rebuilt_index,
        )
    finally:
        reset_rag_service()


@router.post("/docs/upsert", response_model=UpsertDocsResponse)
async def upsert_docs(payload: Annotated[UpsertDocsRequest, Body(...)]) -> UpsertDocsResponse:
    def _upsert_operation() -> docs_service.UpsertDocsSummary:
        return docs_service.upsert_docs_sync(
            docs=payload.docs,
            settings_obj=settings,
            ports=docs_mutation_ports(),
        )

    try:
        summary = cast(
            "docs_service.UpsertDocsSummary",
            await run_blocking(
                run_multi_store_write_locked,
                _upsert_operation,
                task_type="mutation",
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
    except docs_service.TombstonedExternalIdsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        reset_rag_service()

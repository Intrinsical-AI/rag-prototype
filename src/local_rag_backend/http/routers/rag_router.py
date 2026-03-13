"""
Bounded router for core RAG query/evaluation endpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Query

from local_rag_backend.core.use_cases.errors import BadRequestError, InternalServerError
from local_rag_backend.core.use_cases.rag_query import (
    execute_ask_eval_sync,
    list_history_entries_sync,
)
from local_rag_backend.http.dependencies import (
    get_app_container_dependency,
    get_rag_service,
    get_settings_dependency,
)
from local_rag_backend.http.schemas.rag_api_models import (
    AskEvalRequest,
    AskEvalResponse,
    AskRequest,
    AskResponse,
    DocumentInDB,
    HistoryItem,
    QueryResult,
)
from local_rag_backend.infrastructure.concurrency.blocking import run_blocking
from local_rag_backend.infrastructure.observability.observability import (
    Timer,
    fingerprint_question,
    log_event,
    observe_query,
)

if TYPE_CHECKING:
    from local_rag_backend.composition.container import AppContainer
    from local_rag_backend.core.services.rag_runtime import RagService
    from local_rag_backend.settings import Settings

router = APIRouter()


@router.post("/ask", response_model=AskResponse, tags=["RAG"], summary="Ask a question using RAG")
async def ask(
    request: AskRequest,
    service: RagService = Depends(get_rag_service),
    settings_obj: Settings = Depends(get_settings_dependency),
) -> AskResponse:
    """Ask a question using Retrieval-Augmented Generation."""
    t = Timer()
    ok = False
    try:
        rag_result = await run_blocking(service.ask, request.question, request.k)
        ok = True
    finally:
        observe_query(
            retrieval_mode=str(settings_obj.retrieval_mode),
            reranker_enabled=bool(settings_obj.enable_reranker),
            ok=ok,
            duration_s=t.seconds(),
        )

    docs = rag_result["docs"]
    scores = rag_result["scores"]
    log_event(
        "rag_query",
        **fingerprint_question(request.question),
        k=int(request.k),
        retrieval_mode=str(settings_obj.retrieval_mode),
        reranker_enabled=bool(settings_obj.enable_reranker),
        sources=len(docs),
        duration_ms=int(1000 * t.seconds()),
    )

    sources = [
        QueryResult(
            document=DocumentInDB(id=str(doc.id), content=doc.content),
            score=score,
        )
        for doc, score in zip(docs, scores, strict=False)
    ]
    return AskResponse(answer=rag_result["answer"], sources=sources)


@router.get("/history", response_model=list[HistoryItem], tags=["RAG"], summary="Get query history")
async def history(
    limit: int = Query(10, ge=1, le=100, description="Max number of history items to retrieve"),
    offset: int = Query(0, ge=0, description="Number of items to skip (useful for pagination)"),
    container: AppContainer = Depends(get_app_container_dependency),
) -> list[HistoryItem]:
    """Retrieve historical Q&A pairs from the database."""
    history_reader = container.build_history_read_port()
    history_entries = list_history_entries_sync(
        history_reader=history_reader,
        limit=limit,
        offset=offset,
    )

    return [
        HistoryItem(
            id=entry.id,
            question=entry.question,
            answer=entry.answer,
            created_at=(
                entry.created_at.isoformat()
                if hasattr(entry.created_at, "isoformat")
                else str(entry.created_at)
            ),
            source_ids=[str(x) for x in (entry.source_ids or [])],
        )
        for entry in history_entries
    ]


@router.post(
    "/ask_eval",
    response_model=AskEvalResponse,
    tags=["RAG"],
    summary="Ask a question with per-request ephemeral RAG configuration",
)
async def ask_eval(
    payload: AskEvalRequest,
    container: AppContainer = Depends(get_app_container_dependency),
) -> AskEvalResponse:
    """Execute a RAG query with per-request configuration."""
    cfg = payload.config
    if validation_errors := container.validate_rag_config(cfg):
        raise BadRequestError(f"Invalid config: {'; '.join(validation_errors)}")

    try:
        outcome = await run_blocking(
            execute_ask_eval_sync,
            question=payload.question,
            cfg=cfg,
            rag_runtime_factory=container.build_rag_runtime_factory(),
            task_type="eval",
        )
    except ValueError as e:
        raise BadRequestError(str(e)) from e
    except RuntimeError as e:
        raise InternalServerError(str(e)) from e

    rag_result = outcome.rag_result
    docs = rag_result["docs"]
    scores = rag_result["scores"]
    sources = [
        QueryResult(
            document=DocumentInDB(id=str(doc.id), content=doc.content),
            score=score,
        )
        for doc, score in zip(docs, scores, strict=False)
    ]
    return AskEvalResponse(
        answer=rag_result["answer"],
        sources=sources,
        latency_ms=outcome.latency_ms,
    )

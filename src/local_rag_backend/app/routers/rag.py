"""
Bounded router for core RAG query/evaluation endpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Query

from local_rag_backend.app.application.rag import (
    execute_ask_eval_sync,
    list_history_entries_sync,
)
from local_rag_backend.app.blocking import run_blocking
from local_rag_backend.app.composition import DEFAULT_DENSE_BACKEND_MESSAGE
from local_rag_backend.app.dependencies import (
    get_app_container_dependency,
    get_db,
    get_rag_service,
    get_settings_dependency,
)
from local_rag_backend.app.errors import BadRequestError, InternalServerError
from local_rag_backend.app.observability import (
    Timer,
    fingerprint_question,
    log_event,
    observe_query,
)
from local_rag_backend.app.schemas.rag import (
    AskEvalRequest,
    AskEvalResponse,
    AskRequest,
    AskResponse,
    DocumentInDB,
    HistoryItem,
    QueryResult,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.orm import Session

    from local_rag_backend.app.container import AppContainer
    from local_rag_backend.app.schemas.rag import AskEvalConfig
    from local_rag_backend.core.domain.entities import Document as DomainDocument
    from local_rag_backend.core.ports import DocumentRepoPort, GeneratorPort, RetrieverPort
    from local_rag_backend.core.services.rag import RagService
    from local_rag_backend.settings import Settings

router = APIRouter()


def _build_retriever_from_config(
    *,
    container: AppContainer,
    cfg: AskEvalConfig,
    doc_repo: DocumentRepoPort,
    preloaded_docs: Sequence[DomainDocument] | None = None,
) -> RetrieverPort:
    try:
        return container.build_retriever_from_config(
            cfg,
            doc_repo,
            preloaded_docs=preloaded_docs,
            missing_backend_message=DEFAULT_DENSE_BACKEND_MESSAGE,
        )
    except ValueError as e:
        raise BadRequestError(str(e)) from e


def _build_generator_from_config(
    *,
    container: AppContainer,
    cfg: AskEvalConfig,
) -> GeneratorPort:
    try:
        return container.build_generator_from_config(cfg)
    except RuntimeError as e:
        raise InternalServerError(str(e)) from e
    except ValueError as e:
        raise BadRequestError(str(e)) from e


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
    db: Session = Depends(get_db),
) -> list[HistoryItem]:
    """Retrieve historical Q&A pairs from the database."""
    history_entries = list_history_entries_sync(db=db, limit=limit, offset=offset)

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

    outcome = await run_blocking(
        execute_ask_eval_sync,
        question=payload.question,
        cfg=cfg,
        build_retriever_from_config=lambda cfg, doc_repo, *, preloaded_docs=None: (
            _build_retriever_from_config(
                container=container,
                cfg=cfg,
                doc_repo=doc_repo,
                preloaded_docs=preloaded_docs,
            )
        ),
        build_generator_from_config=lambda cfg: _build_generator_from_config(
            container=container,
            cfg=cfg,
        ),
        task_type="eval",
    )

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

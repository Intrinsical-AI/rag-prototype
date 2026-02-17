"""
Bounded router for core RAG query/evaluation endpoints.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from local_rag_backend.app.blocking import run_blocking
from local_rag_backend.app.dependencies import get_rag_service
from local_rag_backend.app.error_mapping import raise_http_for_runtime_error
from local_rag_backend.app.observability import (
    Timer,
    fingerprint_question,
    log_event,
    observe_query,
)
from local_rag_backend.app.schemas import (
    AskEvalRequest,
    AskEvalResponse,
    AskRequest,
    AskResponse,
    DocumentInDB,
    HistoryItem,
    QueryResult,
)
from local_rag_backend.app.wiring import (
    build_generator_from_config,
    build_retriever_from_config,
    validate_rag_config,
)
from local_rag_backend.core.services.rag import RagService
from local_rag_backend.infrastructure.persistence.sqlalchemy.base import get_db
from local_rag_backend.infrastructure.persistence.sqlalchemy.crud import get_history
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import (
    HistorySqlStorage,
    SqlDocumentStorage,
)
from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

router = APIRouter()


@router.post("/ask", response_model=AskResponse, tags=["RAG"], summary="Ask a question using RAG")
async def ask(request: AskRequest, service: RagService = Depends(get_rag_service)) -> AskResponse:
    """Ask a question using Retrieval-Augmented Generation."""
    t = Timer()
    ok = False
    try:
        rag_result = await run_blocking(service.ask, request.question, request.k)
        ok = True
    except Exception as e:
        raise_http_for_runtime_error(e)
        raise
    finally:
        observe_query(ok=ok, duration_s=t.seconds())

    docs = rag_result["docs"]
    scores = rag_result["scores"]
    log_event(
        "rag_query",
        **fingerprint_question(request.question),
        k=int(request.k),
        retrieval_mode=str(settings.retrieval_mode),
        reranker_enabled=bool(settings.enable_reranker),
        sources=len(docs),
        duration_ms=int(1000 * t.seconds()),
    )

    sources = [
        QueryResult(
            document=DocumentInDB(id=doc.id, content=doc.content),
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
    history_entries = get_history(db=db, limit=limit, offset=offset)

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
            source_ids=entry.source_ids or [],
        )
        for entry in history_entries
    ]


@router.post(
    "/ask_eval",
    response_model=AskEvalResponse,
    tags=["RAG"],
    summary="Ask a question with per-request ephemeral RAG configuration",
)
async def ask_eval(payload: AskEvalRequest) -> AskEvalResponse:
    """Execute a RAG query with per-request configuration."""
    cfg = payload.config
    if validation_errors := validate_rag_config(cfg):
        raise HTTPException(status_code=400, detail=f"Invalid config: {'; '.join(validation_errors)}")

    def _run_eval_sync() -> tuple[dict[str, Any], int]:
        doc_repo = SqlDocumentStorage()
        docs = doc_repo.get_all_documents()
        retriever = build_retriever_from_config(cfg, doc_repo, preloaded_docs=docs)
        generator = build_generator_from_config(cfg)

        history_storage = HistorySqlStorage()
        service = RagService(retriever, generator, history_storage)
        t0 = time.perf_counter()
        rag_result = service.ask(question=payload.question, top_k=cfg.k)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return rag_result, latency_ms

    try:
        rag_result, latency_ms = await run_blocking(_run_eval_sync, task_type="eval")
    except Exception as e:
        raise_http_for_runtime_error(e)
        raise

    docs = rag_result["docs"]
    scores = rag_result["scores"]
    sources = [
        QueryResult(
            document=DocumentInDB(id=doc.id, content=doc.content),
            score=score,
        )
        for doc, score in zip(docs, scores, strict=False)
    ]
    return AskEvalResponse(answer=rag_result["answer"], sources=sources, latency_ms=latency_ms)

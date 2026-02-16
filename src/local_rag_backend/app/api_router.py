# src/local_rag_backend/app/api_router.py
"""
FastAPI router for the application endpoints.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import requests
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from openai import OpenAI
from pydantic import BaseModel, Field
from sqlalchemy import text

from local_rag_backend.app.blocking import run_blocking
from local_rag_backend.app.composition import (
    build_dense_embedder_from_settings,
    build_generator_from_settings,
    build_retriever_with_default_embedder_from_settings,
    get_available_llm_providers as get_available_llm_providers_from_settings,
)
from local_rag_backend.app.dependencies import get_rag_service, reset_rag_service
from local_rag_backend.app.diagnostics import (
    get_document_ids,
    get_documents_count,
    get_history_count,
    get_retrieval_index_stats,
)
from local_rag_backend.app.observability import (
    Timer,
    fingerprint_question,
    log_event,
    observe_ingest,
    observe_query,
)
from local_rag_backend.app.schemas import (
    AskEvalConfig,
    AskEvalRequest,
    AskEvalResponse,
    AskRequest,
    AskResponse,
    DeleteDocsByExternalIdRequest,
    DeleteDocsByExternalIdResponse,
    DeleteDocsRequest,
    DeleteDocsResponse,
    DocumentInDB,
    HistoryItem,
    QueryResult,
    RebuildIndexResponse,
    UpsertDocResult,
    UpsertDocsRequest,
    UpsertDocsResponse,
)
from local_rag_backend.app.services import docs as docs_service, index as index_service
from local_rag_backend.app.services.ports import (
    DocsMutationPorts,
    IndexMutationPorts,
)
from local_rag_backend.core.services.dense_upsert import (
    precompute_vectors_for_changed_items,
    sync_dense_after_upsert,
)
from local_rag_backend.core.services.maintenance import (
    delete_documents_multi_store,
    delete_external_ids_multi_store,
    rebuild_index_from_db,
)
from local_rag_backend.core.services.prompting import PromptTemplateError, validate_prompt_template
from local_rag_backend.core.services.rag import RagService
from local_rag_backend.core.services.reranking import RerankingRetriever
from local_rag_backend.core.services.write_lock import multi_store_write_lock
from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder
from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
    SentenceTransformerEmbedder,
)
from local_rag_backend.infrastructure.llms.ollama_chat import OllamaGenerator
from local_rag_backend.infrastructure.llms.openai_chat import OpenAIGenerator
from local_rag_backend.infrastructure.llms.openai_client import create_openai_client
from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
from local_rag_backend.infrastructure.persistence.faiss.manifest import (
    expected_manifest_config_from_settings,
    purge_index_artifacts,
)
from local_rag_backend.infrastructure.persistence.sqlalchemy import base as db_base
from local_rag_backend.infrastructure.persistence.sqlalchemy.base import get_db
from local_rag_backend.infrastructure.persistence.sqlalchemy.crud import get_history
from local_rag_backend.infrastructure.persistence.sqlalchemy.models import Document as DbDocument
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import (
    HistorySqlStorage,
    SqlDocumentStorage,
)
from local_rag_backend.infrastructure.retrieval.dense_faiss import DenseFaissRetriever
from local_rag_backend.infrastructure.retrieval.hybrid import HybridRetriever
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever
from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.orm import Session

    from local_rag_backend.core.domain.entities import Document as DomainDocument
    from local_rag_backend.core.ports import (
        DocumentRepoPort,
        EmbedderPort,
        GeneratorPort,
        RetrieverPort,
    )

# ---------------------- Validation Utilities ---------------------- #


def validate_rag_config(config: AskEvalConfig) -> list[str]:
    """Validate RAG configuration and return list of errors."""
    errors = []
    if config.retrieval_mode not in ["sparse", "dense", "hybrid"]:
        errors.append(f"Invalid retrieval_mode: {config.retrieval_mode}")
    if config.prompt_template is not None:
        try:
            validate_prompt_template(config.prompt_template)
        except PromptTemplateError as e:
            errors.append(f"Invalid prompt_template: {e}")
    return errors


def get_available_llm_providers() -> dict[str, str]:
    """Check available LLM providers based on settings."""
    return get_available_llm_providers_from_settings(settings_obj=settings)


def _build_embedder_for_dense() -> EmbedderPort:
    try:
        return build_dense_embedder_from_settings(
            settings_obj=settings,
            openai_embedder_factory=OpenAIEmbedder,
            st_embedder_factory=lambda model_name: SentenceTransformerEmbedder(model_name=model_name),
            missing_backend_message=(
                "Dense/hybrid operations require an embeddings backend. "
                "Set OPENAI_API_KEY or install the 'dense-st' extra."
            ),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=400,
            detail=(
                "Dense/hybrid operations require an embeddings backend. "
                "Set OPENAI_API_KEY or install the 'dense-st' extra."
            ),
        ) from e


router = APIRouter()


def _run_multi_store_write_locked(fn: Any) -> Any:
    with multi_store_write_lock():
        return fn()


def _docs_mutation_ports() -> DocsMutationPorts:
    return DocsMutationPorts(
        build_embedder=_build_embedder_for_dense,
        doc_repo_factory=cast("Any", lambda: SqlDocumentStorage()),
        build_upsert_doc=SqlDocumentStorage.UpsertDoc,
        vector_repo_factory=FaissVectorStorage,
        precompute_vectors_fn=precompute_vectors_for_changed_items,
        sync_dense_fn=sync_dense_after_upsert,
        rebuild_fn=rebuild_index_from_db,
        delete_docs_fn=delete_documents_multi_store,
        delete_external_ids_fn=delete_external_ids_multi_store,
    )


def _index_mutation_ports() -> IndexMutationPorts:
    return IndexMutationPorts(
        build_embedder=_build_embedder_for_dense,
        doc_repo_factory=lambda: SqlDocumentStorage(),
        vector_repo_factory=FaissVectorStorage,
        purge_index_artifacts_fn=purge_index_artifacts,
        rebuild_fn=rebuild_index_from_db,
    )


# --- Health & Readiness --- #


@router.get("/health", tags=["Health"], summary="Health check endpoint")
async def health_check() -> dict[str, str]:
    """Basic health check for service availability (e.g., Docker/K8s)."""
    try:
        with db_base.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection failed: {e!s}")


@router.get("/ready", tags=["Health"], summary="Readiness check endpoint")
async def readiness_check() -> dict[str, Any]:
    """Check if all dependencies are ready to handle requests."""
    checks: dict[str, Any] = {}
    is_ready = True
    docs_count: int | None = None

    # 1. Database connectivity
    try:
        with db_base.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"failed: {e!s}"
        is_ready = False

    # 1b. Basic DB stats (best-effort)
    if checks.get("database") == "ok":
        try:
            docs_count = get_documents_count(db_base.engine)
            checks["documents"] = {"count": docs_count}
        except Exception as e:
            checks["documents"] = f"failed: {e!s}"
            is_ready = False

        try:
            checks["history"] = {"count": get_history_count(db_base.engine)}
        except Exception as e:
            checks["history"] = f"failed: {e!s}"
            # history is non-critical for answering questions; don't force not-ready here

    # 2. RAG service (best-effort: do not fail the endpoint before reporting readiness)
    try:
        service = await get_rag_service()
        checks["rag_service"] = "ok" if service else "failed: not initialized"
        if not service:
            is_ready = False
    except Exception as e:
        checks["rag_service"] = f"failed: {e!s}"
        is_ready = False

    # 3. LLM providers
    llm_providers = get_available_llm_providers()
    if not llm_providers:
        checks["llm_providers"] = "failed: no providers configured"
        is_ready = False
    else:
        checks["llm_providers"] = llm_providers

    # 4. Retrieval index (for dense/hybrid modes)
    if settings.retrieval_mode in ["dense", "hybrid"]:
        expected_manifest = expected_manifest_config_from_settings(settings)
        stats = get_retrieval_index_stats(
            index_path=settings.index_path,
            id_map_path=settings.id_map_path,
            dim=None,
            expected_manifest=expected_manifest,
        )
        checks["retrieval_index_stats"] = stats

        if stats.get("status") != "ok":
            checks["retrieval_index"] = (
                f"failed: {stats.get('status')} "
                f"(index_path={stats.get('index_path')}, id_map_path={stats.get('id_map_path')}). "
                f"Hint: {stats.get('hint')}"
            )
            is_ready = False
        else:
            checks["retrieval_index"] = "ok"

            # Consistency checks vs SQLite (best-effort, but actionable when it fails)
            if docs_count is not None:
                vectors = int(stats.get("vectors") or 0)
                id_map_len = int(stats.get("id_map_len") or 0)
                if docs_count != id_map_len:
                    checks["retrieval_index"] = (
                        f"failed: drift detected (documents={docs_count}, vectors={vectors}). "
                        "Hint: rebuild the index (`rag-rebuild-index` or POST /api/index/rebuild)."
                    )
                    is_ready = False
                else:
                    # Optional deep check: compare sets when the corpus is small enough.
                    # Keep /ready fast for larger corpora.
                    if docs_count <= 5000:
                        try:
                            db_ids = set(get_document_ids(db_base.engine))
                            index_ids = set(
                                json.loads(Path(settings.id_map_path).read_text(encoding="utf-8"))
                            )
                            stale = sorted(index_ids - db_ids)
                            missing = sorted(db_ids - index_ids)
                            if stale or missing:
                                checks["retrieval_index_drift"] = {
                                    "stale_in_index": stale[:20],
                                    "missing_in_index": missing[:20],
                                    "stale_count": len(stale),
                                    "missing_count": len(missing),
                                }
                                checks["retrieval_index"] = (
                                    "failed: drift detected (ID set mismatch). "
                                    "Hint: rebuild the index (`rag-rebuild-index` or POST /api/index/rebuild)."
                                )
                                is_ready = False
                        except Exception as e:
                            checks["retrieval_index_drift"] = f"failed: {e!s}"

    response_payload = {"status": "ready" if is_ready else "not_ready", "checks": checks}
    if not is_ready:
        raise HTTPException(status_code=503, detail=response_payload)
    return response_payload


@router.get("/health/ollama", tags=["Health"], summary="Ollama server health check")
async def ollama_health_check() -> dict[str, Any]:
    """
    Checks if the Ollama server is running and accessible.

    Returns:
        dict: A dictionary with the status of the Ollama server.
    """
    try:
        response = await run_blocking(
            requests.get, settings.ollama_base_url, timeout=5, task_type="network"
        )
        response.raise_for_status()
        return {"status": "ok", "url": settings.ollama_base_url}
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama server not accessible at {settings.ollama_base_url}. Error: {e}",
        )


# ---------------------- API Endpoints ---------------------- #


@router.post("/ask", response_model=AskResponse, tags=["RAG"], summary="Ask a question using RAG")
async def ask(request: AskRequest, service: RagService = Depends(get_rag_service)) -> AskResponse:
    """
    Ask a question using Retrieval-Augmented Generation.
    Args:
        request: Question and retrieval parameters
    Returns:
        AskResponse: Generated answer with source documents
    """
    # Avoid blocking the event loop: the RAG pipeline is synchronous (DB/FAISS + network I/O).
    t = Timer()
    ok = False
    try:
        rag_result = await run_blocking(service.ask, request.question, request.k)
        ok = True
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
    """
    Retrieves historical Q&A pairs from the database.
    Args:
        limit (int): Maximum number of history records to return.
        offset (int): Number of records to skip.
        db (Session): Database session dependency.
    Returns:
        List[HistoryItem]: List of historical Q&A entries.
    """
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


# --- DOCS API (minimal KB) ---


@router.get("/docs", response_model=list[DocumentInDB])
async def list_docs(
    limit: int = Query(100, ge=1, le=1000, description="Max number of docs"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
) -> list[DocumentInDB]:
    docs = db.query(DbDocument).order_by(DbDocument.id.asc()).offset(offset).limit(limit).all()
    return [DocumentInDB.model_validate(d) for d in docs]


class IngestRequest(BaseModel):
    texts: list[Annotated[str, Field(max_length=20000)]] = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Raw texts to ingest (max 64 items, 20k chars each)",
    )


class IngestResponse(BaseModel):
    count: int
    ids: list[int]


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
            ports=_docs_mutation_ports(),
        )

    ok = False
    ids: list[int] = []
    try:
        ids = cast(
            "list[int]",
            await run_blocking(
                _run_multi_store_write_locked, _ingest_operation, task_type="mutation"
            ),
        )
        ok = True
        return IngestResponse(count=len(ids), ids=ids)
    except RuntimeError as e:
        if "sentence-transformers" in str(e) or "Dense/hybrid" in str(e):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Dense/hybrid ingestion requires an embeddings backend. "
                    "Set OPENAI_API_KEY or install the 'dense-st' extra."
                ),
            ) from e
        raise
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
        # Always bust cache after a mutating attempt.
        # Even failed writes can be partially applied in degraded scenarios (e.g. rebuild failure).
        reset_rag_service()


@router.post("/docs/delete_by_external_id", response_model=DeleteDocsByExternalIdResponse)
async def delete_docs_by_external_id(
    payload: Annotated[DeleteDocsByExternalIdRequest, Body(...)],
) -> DeleteDocsByExternalIdResponse:
    def _delete_operation() -> docs_service.DeleteDocsByExternalIdSummary:
        return docs_service.delete_docs_by_external_id_sync(
            external_ids=payload.external_ids,
            settings_obj=settings,
            ports=_docs_mutation_ports(),
        )

    try:
        summary = cast(
            "docs_service.DeleteDocsByExternalIdSummary",
            await run_blocking(
                _run_multi_store_write_locked, _delete_operation, task_type="mutation"
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
            ports=_docs_mutation_ports(),
        )

    try:
        summary = cast(
            "docs_service.DeleteDocsSummary",
            await run_blocking(
                _run_multi_store_write_locked, _delete_operation, task_type="mutation"
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
            ports=_docs_mutation_ports(),
        )

    try:
        summary = cast(
            "docs_service.UpsertDocsSummary",
            await run_blocking(
                _run_multi_store_write_locked, _upsert_operation, task_type="mutation"
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


@router.post("/index/rebuild", response_model=RebuildIndexResponse)
async def rebuild_index() -> RebuildIndexResponse:
    if settings.retrieval_mode not in ("dense", "hybrid"):
        raise HTTPException(status_code=400, detail="Index rebuild requires dense or hybrid mode.")

    def _rebuild_operation() -> int:
        return index_service.rebuild_index_sync(
            settings_obj=settings,
            ports=_index_mutation_ports(),
        )

    try:
        indexed = cast(
            "int",
            await run_blocking(
                _run_multi_store_write_locked, _rebuild_operation, task_type="mutation"
            ),
        )
        return RebuildIndexResponse(indexed=indexed)
    finally:
        reset_rag_service()


def _build_retriever_from_config(
    cfg: AskEvalConfig,
    doc_repo: DocumentRepoPort,
    *,
    preloaded_docs: Sequence[DomainDocument] | None = None,
) -> RetrieverPort:
    """Build a retriever instance based on dynamic configuration."""
    try:
        return build_retriever_with_default_embedder_from_settings(
            settings_obj=settings,
            retrieval_mode=cfg.retrieval_mode,
            doc_repo=doc_repo,
            openai_embedder_factory=OpenAIEmbedder,
            st_embedder_factory=lambda model_name: SentenceTransformerEmbedder(model_name=model_name),
            missing_backend_message=(
                "Dense/hybrid operations require an embeddings backend. "
                "Set OPENAI_API_KEY or install the 'dense-st' extra."
            ),
            preloaded_docs=preloaded_docs,
            hybrid_alpha=cfg.hybrid_alpha,
            enable_reranker=settings.enable_reranker,
            reranker_candidate_k=settings.reranker_candidate_k,
            reranker_strategy=settings.reranker_strategy,
            sparse_retriever_factory=SparseBM25Retriever,
            dense_retriever_factory=DenseFaissRetriever,
            hybrid_retriever_factory=HybridRetriever,
            vector_repo_factory=FaissVectorStorage,
            reranker_factory=RerankingRetriever,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _build_generator_from_config(cfg: AskEvalConfig) -> GeneratorPort:
    """Build a generator instance based on dynamic configuration."""
    available_providers = get_available_llm_providers()
    try:
        return build_generator_from_settings(
            settings_obj=settings,
            llm_provider=cfg.llm_provider,
            model=cfg.model,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            prompt_template=cfg.prompt_template,
            openai_generator_factory=OpenAIGenerator,
            ollama_generator_factory=OllamaGenerator,
            available_providers=available_providers,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post(
    "/ask_eval",
    response_model=AskEvalResponse,
    tags=["RAG"],
    summary="Ask a question with per-request ephemeral RAG configuration",
)
async def ask_eval(payload: AskEvalRequest) -> AskEvalResponse:
    """Execute a RAG query with per-request configuration."""
    cfg = payload.config

    # Validate configuration
    if validation_errors := validate_rag_config(cfg):
        raise HTTPException(
            status_code=400, detail=f"Invalid config: {'; '.join(validation_errors)}"
        )

    def _run_eval_sync() -> tuple[dict[str, Any], int]:
        doc_repo = SqlDocumentStorage()
        docs = doc_repo.get_all_documents()
        retriever = _build_retriever_from_config(cfg, doc_repo, preloaded_docs=docs)
        generator = _build_generator_from_config(cfg)

        history_storage = HistorySqlStorage()
        service = RagService(retriever, generator, history_storage)
        t0 = time.perf_counter()
        rag_result = service.ask(question=payload.question, top_k=cfg.k)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return rag_result, latency_ms

    rag_result, latency_ms = await run_blocking(_run_eval_sync, task_type="eval")
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


# ---------------------- OpenRouter Proxy (CodeArena) ---------------------- #


class OpenRouterUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenRouterGenerateRequest(BaseModel):
    model: str | None = Field(
        default=None, description="OpenRouter model ID, e.g., 'openai/gpt-4o-mini'"
    )
    system_instruction: str = Field(..., min_length=1, max_length=8000)
    user_content: str = Field(..., min_length=1, max_length=8000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)


class OpenRouterGenerateResponse(BaseModel):
    text: str
    usage: OpenRouterUsage | None = None


@router.post(
    "/openrouter/generate",
    response_model=OpenRouterGenerateResponse,
    tags=["LLM"],
    summary="Proxy completion via OpenRouter (OpenAI-compatible)",
)
async def openrouter_generate(payload: OpenRouterGenerateRequest) -> OpenRouterGenerateResponse:
    if not (
        getattr(settings, "openrouter_enabled", False)
        and getattr(settings, "openrouter_api_key", None)
    ):
        raise HTTPException(
            status_code=400,
            detail="OpenRouter is not configured (set OPENROUTER_ENABLED and OPENROUTER_API_KEY)",
        )

    headers: dict[str, str] = {}
    if settings.openrouter_site_url is not None:
        headers["HTTP-Referer"] = settings.openrouter_site_url
    if settings.openrouter_app_title is not None:
        headers["X-Title"] = settings.openrouter_app_title

    def _create_sync() -> Any:
        client = create_openai_client(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            default_headers=headers or None,
            timeout=settings.openai_request_timeout,
            client_factory=OpenAI,
        )
        return client.chat.completions.create(
            model=(payload.model or settings.openrouter_model),
            temperature=payload.temperature,
            top_p=payload.top_p,
            max_tokens=payload.max_tokens,
            messages=[
                {"role": "system", "content": payload.system_instruction},
                {"role": "user", "content": payload.user_content},
            ],
        )

    try:
        resp = await run_blocking(_create_sync, task_type="network")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenRouter error: {e!s}") from e

    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    usage_obj = None
    if usage is not None:
        usage_obj = OpenRouterUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )

    return OpenRouterGenerateResponse(text=text, usage=usage_obj)


# ---------------------- Templates API ---------------------- #


class TemplateResponse(BaseModel):
    name: str
    template: str
    description: str


@router.get(
    "/templates",
    response_model=list[TemplateResponse],
    tags=["RAG"],
    summary="Get available prompt templates",
)
async def get_templates() -> list[TemplateResponse]:
    """
    Get available prompt templates for RAG queries.
    Returns:
        List[TemplateResponse]: Available prompt templates with descriptions
    """
    templates = [
        TemplateResponse(
            name="default",
            template=settings.openai_prompt_template,
            description="Default template for OpenAI/OpenRouter models",
        ),
        TemplateResponse(
            name="ollama",
            template=settings.ollama_prompt_template,
            description="Template optimized for Ollama models",
        ),
        TemplateResponse(
            name="concise",
            template="Based on the context below, provide a concise answer.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:",
            description="Concise template for brief responses",
        ),
        TemplateResponse(
            name="detailed",
            template="You are an expert Q&A system. Your task is to answer the user's question based on the provided sources. Synthesize the information from the sources into a coherent, detailed answer.\n\nSources:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            description="Detailed template for comprehensive responses",
        ),
    ]
    return templates


# ---------------------- Configuration API ---------------------- #


class ConfigResponse(BaseModel):
    retrieval_mode: str
    hybrid_alpha: float
    temperature: float
    max_tokens: int
    available_providers: list[str]


@router.get(
    "/config",
    response_model=ConfigResponse,
    tags=["RAG"],
    summary="Get backend configuration defaults",
)
async def get_config() -> ConfigResponse:
    """
    Get backend configuration defaults.
    Returns:
        ConfigResponse: Current backend configuration defaults
    """
    available_providers = list(get_available_llm_providers().keys())

    return ConfigResponse(
        retrieval_mode=settings.retrieval_mode,
        hybrid_alpha=settings.hybrid_retrieval_alpha,
        temperature=settings.openai_temperature,
        max_tokens=settings.openai_max_tokens,
        available_providers=available_providers,
    )

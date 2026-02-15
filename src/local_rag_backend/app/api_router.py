# src/local_rag_backend/app/api_router.py
"""
FastAPI router for the application endpoints.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import requests
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from openai import OpenAI
from pydantic import BaseModel, Field
from sqlalchemy import text

from local_rag_backend.app.dependencies import get_rag_service
from local_rag_backend.core.services.etl import ETLService
from local_rag_backend.core.services.rag import RagService
from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder
from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
    SentenceTransformerEmbedder,
)
from local_rag_backend.infrastructure.llms.ollama_chat import OllamaGenerator
from local_rag_backend.infrastructure.llms.openai_chat import OpenAIGenerator
from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
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
from local_rag_backend.models import (
    AskEvalConfig,
    AskEvalRequest,
    AskEvalResponse,
    AskRequest,
    AskResponse,
    DocumentInDB,
    HistoryItem,
    QueryResult,
)
from local_rag_backend.settings import settings
from local_rag_backend.utils import get_corpus_and_ids

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

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
    if not (1 <= config.k <= 10):
        errors.append(f"Invalid k value: {config.k}. Must be between 1 and 10")
    if config.retrieval_mode == "hybrid" and not (0.0 <= (config.hybrid_alpha or 0.5) <= 1.0):
        errors.append(f"Invalid hybrid_alpha: {config.hybrid_alpha}. Must be between 0.0 and 1.0")
    if config.temperature is not None and not (0.0 <= config.temperature <= 2.0):
        errors.append(f"Invalid temperature: {config.temperature}. Must be between 0.0 and 2.0")
    if config.max_tokens is not None and not (1 <= config.max_tokens <= 4096):
        errors.append(f"Invalid max_tokens: {config.max_tokens}. Must be between 1 and 4096")
    return errors


def get_available_llm_providers() -> dict[str, str]:
    """Check available LLM providers based on settings."""
    providers = {}
    if settings.openai_api_key:
        providers["openai"] = "configured"
    if settings.ollama_enabled:
        providers["ollama"] = "enabled"
    if getattr(settings, "openrouter_enabled", False) and getattr(
        settings, "openrouter_api_key", None
    ):
        providers["openrouter"] = "configured"
    return providers


router = APIRouter()

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
async def readiness_check(service: RagService = Depends(get_rag_service)) -> dict[str, Any]:
    """Check if all dependencies are ready to handle requests."""
    checks: dict[str, Any] = {}
    is_ready = True

    # 1. Database connectivity
    try:
        with db_base.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"failed: {e!s}"
        is_ready = False

    # 2. RAG service
    checks["rag_service"] = "ok" if service else "failed: not initialized"
    if not service:
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
        checks["retrieval_index"] = (
            "ok" if Path(settings.index_path).exists() else "warning: not found"
        )

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
        response = await asyncio.to_thread(requests.get, settings.ollama_base_url, timeout=5)
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
    rag_result = service.ask(question=request.question, top_k=request.k)
    docs = rag_result["docs"]
    scores = rag_result["scores"]

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
    texts: list[str] = Field(..., min_length=1, description="Raw texts to ingest")


class IngestResponse(BaseModel):
    count: int
    ids: list[int]


@router.post("/docs", response_model=IngestResponse)
async def ingest_docs(payload: Annotated[IngestRequest, Body(...)]) -> IngestResponse:
    texts = [t.strip() for t in payload.texts if t and t.strip()]
    if not texts:
        return IngestResponse(count=0, ids=[])

    # Repos
    doc_repo = SqlDocumentStorage()

    if settings.retrieval_mode in ("dense", "hybrid"):
        embedder: EmbedderPort = (
            OpenAIEmbedder()
            if settings.openai_api_key
            else SentenceTransformerEmbedder(model_name=settings.st_embedding_model)
        )
        vec = FaissVectorStorage(
            index_path=settings.index_path,
            id_map_path=settings.id_map_path,
            dim=embedder.dim,
        )
        etl = ETLService(doc_repo, vec, embedder)
        ids = list(etl.ingest(texts))
    else:
        ids = list(doc_repo.store_documents(texts))

    return IngestResponse(count=len(ids), ids=ids)


def _build_retriever_from_config(
    cfg: AskEvalConfig,
    doc_repo: DocumentRepoPort,
    corpus: list[str],
    doc_ids: list[int],
) -> RetrieverPort:
    """Build a retriever instance based on dynamic configuration."""
    if cfg.retrieval_mode == "sparse":
        return SparseBM25Retriever(documents=corpus, doc_ids=doc_ids, doc_repo=doc_repo)

    embedder: EmbedderPort = (
        OpenAIEmbedder()
        if settings.openai_api_key
        else SentenceTransformerEmbedder(model_name=settings.st_embedding_model)
    )
    faiss_storage = FaissVectorStorage(
        index_path=settings.index_path, id_map_path=settings.id_map_path, dim=embedder.dim
    )
    dense_retriever = DenseFaissRetriever(
        embedder=embedder, faiss_index=faiss_storage, doc_repo=doc_repo
    )

    if cfg.retrieval_mode == "dense":
        return dense_retriever

    if cfg.retrieval_mode == "hybrid":
        sparse_retriever = SparseBM25Retriever(documents=corpus, doc_ids=doc_ids, doc_repo=doc_repo)
        alpha = (
            cfg.hybrid_alpha if cfg.hybrid_alpha is not None else settings.hybrid_retrieval_alpha
        )
        return HybridRetriever(dense=dense_retriever, sparse=sparse_retriever, alpha=alpha)

    raise HTTPException(status_code=400, detail=f"Unsupported retrieval_mode: {cfg.retrieval_mode}")


def _build_generator_from_config(cfg: AskEvalConfig) -> GeneratorPort:
    """Build a generator instance based on dynamic configuration."""
    available_providers = get_available_llm_providers()
    provider = cfg.llm_provider or next(iter(available_providers), None)

    if not provider:
        raise HTTPException(status_code=500, detail="No LLM provider available.")

    if provider not in available_providers:
        raise HTTPException(
            status_code=400, detail=f"LLM provider '{provider}' is not available or configured."
        )

    # Note: use explicit keyword args to satisfy static typing

    if provider == "openrouter":
        headers: dict[str, str] = {}
        if settings.openrouter_site_url is not None:
            headers["HTTP-Referer"] = settings.openrouter_site_url
        if settings.openrouter_app_title is not None:
            headers["X-Title"] = settings.openrouter_app_title
        return OpenAIGenerator(
            model=(cfg.model or settings.openrouter_model),
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            prompt_template=cfg.prompt_template,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            extra_headers=headers or None,
        )
    if provider == "openai":
        return OpenAIGenerator(
            model=cfg.model,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            prompt_template=cfg.prompt_template,
        )
    if provider == "ollama":
        return OllamaGenerator(
            model=cfg.model, temperature=cfg.temperature, prompt_template=cfg.prompt_template
        )

    raise HTTPException(status_code=400, detail=f"Unsupported llm_provider: {provider}")


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

    # Build components on the fly
    doc_repo = SqlDocumentStorage()
    corpus, doc_ids = get_corpus_and_ids(doc_repo)
    retriever = _build_retriever_from_config(cfg, doc_repo, corpus, doc_ids)
    generator = _build_generator_from_config(cfg)

    # Execute RAG ask
    history_storage = HistorySqlStorage()
    service = RagService(retriever, generator, history_storage)
    t0 = time.perf_counter()
    rag_result = service.ask(question=payload.question, top_k=cfg.k)
    latency_ms = int((time.perf_counter() - t0) * 1000)
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
    system_instruction: str
    user_content: str
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None


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

    client = OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        default_headers=headers or None,
    )

    try:

        def _create() -> Any:
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

        resp = await asyncio.to_thread(_create)
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
    available_providers = []
    if settings.openai_api_key:
        available_providers.append("openai")
    if settings.ollama_enabled:
        available_providers.append("ollama")
    if getattr(settings, "openrouter_enabled", False) and getattr(
        settings, "openrouter_api_key", None
    ):
        available_providers.append("openrouter")

    return ConfigResponse(
        retrieval_mode=settings.retrieval_mode,
        hybrid_alpha=settings.hybrid_retrieval_alpha,
        temperature=settings.openai_temperature,
        max_tokens=settings.openai_max_tokens,
        available_providers=available_providers,
    )

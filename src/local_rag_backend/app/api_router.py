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
from local_rag_backend.core.services.corpus import get_corpus_and_ids
from local_rag_backend.core.services.dense_upsert import (
    precompute_vectors_for_changed_items,
    sync_dense_after_upsert,
)
from local_rag_backend.core.services.maintenance import (
    delete_documents_multi_store,
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
    if config.prompt_template is not None:
        try:
            validate_prompt_template(config.prompt_template)
        except PromptTemplateError as e:
            errors.append(f"Invalid prompt_template: {e}")
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


def _build_embedder_for_dense() -> EmbedderPort:
    if settings.openai_api_key:
        return OpenAIEmbedder()
    try:
        return SentenceTransformerEmbedder(model_name=settings.st_embedding_model)
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
        response = await run_blocking(requests.get, settings.ollama_base_url, timeout=5)
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

    def _embedding_model_name_for_dedup() -> str:
        if settings.retrieval_mode not in ("dense", "hybrid"):
            return "none"
        if settings.openai_api_key:
            return str(settings.openai_embedding_model)
        return str(settings.st_embedding_model)

    def _ingest_sync() -> list[int]:
        from local_rag_backend.core.services.chunking import chunk_chars_v1
        from local_rag_backend.core.services.dedup import chunk_dedup_sha256
        from local_rag_backend.core.services.ingestion import (
            build_preprocess_fn_from_settings,
            default_formatter,
        )

        doc_repo = SqlDocumentStorage()
        preprocess_fn = build_preprocess_fn_from_settings(settings)

        chunker_version = str(settings.ingest_chunker_version)
        embed_model = _embedding_model_name_for_dedup()

        # Build unique items by external_id (dedup hash) and keep a stable insertion order.
        source_id = f"api:/docs:v={chunker_version}:emb={embed_model}"
        unique_extids: list[str] = []
        items_by_extid: dict[str, SqlDocumentStorage.UpsertDoc] = {}

        for i, raw in enumerate(texts):
            md_base: dict[str, object] = {"source": "api:/docs", "input_index": i}
            processed = preprocess_fn(raw, md_base)
            chunks = chunk_chars_v1(
                processed,
                max_chars=settings.ingest_chunk_chars,
                overlap=settings.ingest_chunk_overlap,
            )
            for c in chunks:
                dedup = chunk_dedup_sha256(
                    cleaned_text=c.text,
                    chunker_version=chunker_version,
                    embedding_model_name=embed_model,
                )
                external_id = f"chunk:{dedup}"

                if external_id in items_by_extid:
                    continue
                unique_extids.append(external_id)

                md = dict(md_base)
                md["chunk_index"] = int(c.chunk_index)
                md["chunk_start_char"] = int(c.start_char)
                md["chunk_end_char"] = int(c.end_char)
                md["chunker_version"] = chunker_version
                md["embedding_model"] = embed_model
                md["dedup_sha256"] = dedup
                md["parent_doc_id"] = f"api:/docs:text={i}"

                content = default_formatter(c.text, md)
                items_by_extid[external_id] = SqlDocumentStorage.UpsertDoc(
                    external_id=external_id,
                    content=content,
                    source_id=source_id,
                    metadata=md,
                    chunk_dedup_sha256=dedup,
                )

        unique_items = list(items_by_extid.values())
        tombstoned = doc_repo.get_tombstoned_external_ids(unique_extids)
        if tombstoned:
            unique_extids = [e for e in unique_extids if e not in tombstoned]
            unique_items = [it for it in unique_items if it.external_id not in tombstoned]
        if not unique_items:
            return []

        embedder = None
        vectors_by_external_id: dict[str, list[float]] = {}
        if settings.retrieval_mode in ("dense", "hybrid"):
            # Compute embeddings before SQL upsert so provider failures don't leave SQL/index drift.
            embedder = _build_embedder_for_dense()
            vectors_by_external_id = precompute_vectors_for_changed_items(
                items=unique_items,
                doc_repo=doc_repo,
                embedder=embedder,
            )

        results, _changed_content, updated_content_ids = doc_repo.upsert_documents_by_external_id(
            unique_items
        )
        id_by_ext = {r.external_id: int(r.id) for r in results}

        if settings.retrieval_mode in ("dense", "hybrid") and embedder is not None:
            vec = FaissVectorStorage(
                index_path=settings.index_path,
                id_map_path=settings.id_map_path,
                dim=embedder.dim,
            )
            sync_dense_after_upsert(
                results=results,
                updated_content_ids=updated_content_ids,
                vectors_by_external_id=vectors_by_external_id,
                vec_repo=vec,
                doc_repo=doc_repo,
                embedder=embedder,
                rebuild_fn=rebuild_index_from_db,
            )

        return [id_by_ext[e] for e in unique_extids if e in id_by_ext]

    ok = False
    ids: list[int] = []
    try:
        ids = cast("list[int]", await run_blocking(_run_multi_store_write_locked, _ingest_sync))
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
    external_ids = [str(x).strip() for x in payload.external_ids if str(x).strip()]
    if not external_ids:
        return DeleteDocsByExternalIdResponse(
            deleted_sql=0,
            deleted_index=0,
            tombstoned=0,
            missing_external_ids=[],
            rebuilt_index=False,
        )

    def _delete_sync() -> DeleteDocsByExternalIdResponse:
        doc_repo = SqlDocumentStorage()
        deleted_sql, deleted_ids, missing, tombstoned = doc_repo.delete_by_external_ids(
            external_ids
        )

        if settings.retrieval_mode in ("dense", "hybrid"):
            vec = FaissVectorStorage(
                index_path=settings.index_path,
                id_map_path=settings.id_map_path,
                dim=None,
            )
            try:
                deleted_index = int(vec.delete(deleted_ids))
                return DeleteDocsByExternalIdResponse(
                    deleted_sql=deleted_sql,
                    deleted_index=deleted_index,
                    tombstoned=tombstoned,
                    missing_external_ids=missing,
                    rebuilt_index=False,
                )
            except Exception:
                embedder = _build_embedder_for_dense()
                rebuilt_n = rebuild_index_from_db(
                    doc_repo=doc_repo, vec_repo=vec, embedder=embedder
                )
                return DeleteDocsByExternalIdResponse(
                    deleted_sql=deleted_sql,
                    deleted_index=None,
                    tombstoned=tombstoned,
                    missing_external_ids=missing,
                    rebuilt_index=rebuilt_n >= 0,
                )

        return DeleteDocsByExternalIdResponse(
            deleted_sql=deleted_sql,
            deleted_index=None,
            tombstoned=tombstoned,
            missing_external_ids=missing,
            rebuilt_index=False,
        )

    try:
        return cast(
            "DeleteDocsByExternalIdResponse",
            await run_blocking(_run_multi_store_write_locked, _delete_sync),
        )
    finally:
        reset_rag_service()


@router.post("/docs/delete", response_model=DeleteDocsResponse)
async def delete_docs(payload: Annotated[DeleteDocsRequest, Body(...)]) -> DeleteDocsResponse:
    ids = [int(i) for i in payload.ids]
    if not ids:
        return DeleteDocsResponse(deleted_sql=0, deleted_index=0, rebuilt_index=False)

    def _delete_sync() -> DeleteDocsResponse:
        doc_repo = SqlDocumentStorage()
        if settings.retrieval_mode in ("dense", "hybrid"):
            vec = FaissVectorStorage(
                index_path=settings.index_path,
                id_map_path=settings.id_map_path,
                dim=None,  # infer from existing index when possible
            )
            deleted_sql, deleted_index, rebuilt = delete_documents_multi_store(
                doc_repo=doc_repo,
                vec_repo=vec,
                embedder_factory=_build_embedder_for_dense,
                ids=ids,
                rebuild_on_index_failure=True,
            )
            return DeleteDocsResponse(
                deleted_sql=deleted_sql, deleted_index=deleted_index, rebuilt_index=rebuilt
            )

        deleted_sql, _, _ = delete_documents_multi_store(doc_repo=doc_repo, ids=ids)
        return DeleteDocsResponse(deleted_sql=deleted_sql, deleted_index=None, rebuilt_index=False)

    try:
        return cast(
            "DeleteDocsResponse", await run_blocking(_run_multi_store_write_locked, _delete_sync)
        )
    finally:
        reset_rag_service()


@router.post("/docs/upsert", response_model=UpsertDocsResponse)
async def upsert_docs(payload: Annotated[UpsertDocsRequest, Body(...)]) -> UpsertDocsResponse:
    # Validate uniqueness early for deterministic behavior.
    ext_ids = [d.external_id for d in payload.docs]
    if len(set(ext_ids)) != len(ext_ids):
        raise HTTPException(
            status_code=400, detail="external_id values must be unique per request."
        )

    def _upsert_sync() -> UpsertDocsResponse:
        doc_repo = SqlDocumentStorage()
        tombstoned = doc_repo.get_tombstoned_external_ids([d.external_id for d in payload.docs])
        if tombstoned:
            raise HTTPException(
                status_code=409,
                detail=f"Some external_id values are tombstoned (deleted): {sorted(tombstoned)[:10]}",
            )
        items = [
            SqlDocumentStorage.UpsertDoc(
                external_id=d.external_id,
                content=d.content,
                source_id=d.source_id,
                metadata=d.metadata,
            )
            for d in payload.docs
        ]

        embedder = None
        vectors_by_external_id: dict[str, list[float]] = {}
        if settings.retrieval_mode in ("dense", "hybrid"):
            # Compute embeddings before SQL upsert so provider failures don't leave SQL/index drift.
            embedder = _build_embedder_for_dense()
            vectors_by_external_id = precompute_vectors_for_changed_items(
                items=items,
                doc_repo=doc_repo,
                embedder=embedder,
            )

        results, _changed_content, updated_content_ids = doc_repo.upsert_documents_by_external_id(
            items
        )
        inserted = sum(1 for r in results if r.action == "inserted")
        updated = sum(1 for r in results if r.action == "updated")
        unchanged = sum(1 for r in results if r.action == "unchanged")

        rebuilt_index = False
        if settings.retrieval_mode in ("dense", "hybrid") and embedder is not None:
            vec = FaissVectorStorage(
                index_path=settings.index_path,
                id_map_path=settings.id_map_path,
                dim=embedder.dim,
            )
            rebuilt_index = sync_dense_after_upsert(
                results=results,
                updated_content_ids=updated_content_ids,
                vectors_by_external_id=vectors_by_external_id,
                vec_repo=vec,
                doc_repo=doc_repo,
                embedder=embedder,
                rebuild_fn=rebuild_index_from_db,
            )

        return UpsertDocsResponse(
            inserted=inserted,
            updated=updated,
            unchanged=unchanged,
            rebuilt_index=rebuilt_index,
            results=[
                UpsertDocResult(
                    external_id=r.external_id,
                    id=r.id,
                    action=r.action,
                    content_changed=r.content_changed,
                )
                for r in results
            ],
        )

    try:
        return cast(
            "UpsertDocsResponse", await run_blocking(_run_multi_store_write_locked, _upsert_sync)
        )
    finally:
        reset_rag_service()


@router.post("/index/rebuild", response_model=RebuildIndexResponse)
async def rebuild_index() -> RebuildIndexResponse:
    if settings.retrieval_mode not in ("dense", "hybrid"):
        raise HTTPException(status_code=400, detail="Index rebuild requires dense or hybrid mode.")

    def _rebuild_sync() -> RebuildIndexResponse:
        doc_repo = SqlDocumentStorage()
        embedder = _build_embedder_for_dense()
        # Rebuild must be able to recover from an incompatible on-disk index (e.g. dim drift).
        from local_rag_backend.infrastructure.persistence.faiss.manifest import (
            purge_index_artifacts,
        )

        purge_index_artifacts(index_path=settings.index_path, id_map_path=settings.id_map_path)
        vec = FaissVectorStorage(
            index_path=settings.index_path,
            id_map_path=settings.id_map_path,
            dim=embedder.dim,
        )
        n = rebuild_index_from_db(doc_repo=doc_repo, vec_repo=vec, embedder=embedder)
        return RebuildIndexResponse(indexed=n)

    try:
        return cast(
            "RebuildIndexResponse", await run_blocking(_run_multi_store_write_locked, _rebuild_sync)
        )
    finally:
        reset_rag_service()


def _build_retriever_from_config(
    cfg: AskEvalConfig,
    doc_repo: DocumentRepoPort,
    corpus: list[str],
    doc_ids: list[int],
) -> RetrieverPort:
    """Build a retriever instance based on dynamic configuration."""
    retriever: RetrieverPort
    if cfg.retrieval_mode == "sparse":
        retriever = SparseBM25Retriever(documents=corpus, doc_ids=doc_ids, doc_repo=doc_repo)
        if settings.enable_reranker:
            retriever = RerankingRetriever(
                retriever,
                candidate_k=settings.reranker_candidate_k,
                strategy=settings.reranker_strategy,
            )
        return retriever

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
        retriever = dense_retriever
    elif cfg.retrieval_mode == "hybrid":
        sparse_retriever = SparseBM25Retriever(documents=corpus, doc_ids=doc_ids, doc_repo=doc_repo)
        alpha = (
            cfg.hybrid_alpha if cfg.hybrid_alpha is not None else settings.hybrid_retrieval_alpha
        )
        retriever = HybridRetriever(dense=dense_retriever, sparse=sparse_retriever, alpha=alpha)
    else:
        raise HTTPException(
            status_code=400, detail=f"Unsupported retrieval_mode: {cfg.retrieval_mode}"
        )

    if settings.enable_reranker:
        retriever = RerankingRetriever(
            retriever,
            candidate_k=settings.reranker_candidate_k,
            strategy=settings.reranker_strategy,
        )

    return retriever


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

    def _run_eval_sync() -> tuple[dict[str, Any], int]:
        doc_repo = SqlDocumentStorage()
        corpus, doc_ids = get_corpus_and_ids(doc_repo)
        retriever = _build_retriever_from_config(cfg, doc_repo, corpus, doc_ids)
        generator = _build_generator_from_config(cfg)

        history_storage = HistorySqlStorage()
        service = RagService(retriever, generator, history_storage)
        t0 = time.perf_counter()
        rag_result = service.ask(question=payload.question, top_k=cfg.k)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return rag_result, latency_ms

    rag_result, latency_ms = await run_blocking(_run_eval_sync)
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
        resp = await run_blocking(_create_sync)
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

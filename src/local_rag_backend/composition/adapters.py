"""
Shared composition helpers for adapter selection.

This module centralizes policy decisions for:
- dense embedder selection
- retriever wiring
- generator provider wiring
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

from openai import OpenAI
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from local_rag_backend.core.errors import EmbeddingsBackendUnavailableError
from local_rag_backend.core.ports import (
    BlockingExecutorPort,
    BlockingTaskType,
    DocsImportLoaderPort,
    DocsReadPort,
    EvalDatasetDocInput,
    EvalRetrieverFactoryPort,
    EvalRetrieverPort,
    EvalStoragePort,
    HealthDiagnosticsPort,
    HistoryEntry,
    HistoryReadPort,
    ImportDocsLoadResult,
    ListedDocument,
    OpenRouterClientPort,
    OpenRouterGenerateRequest,
    OpenRouterGenerateResult,
    OpenRouterUsage,
    RagRuntimeFactoryPort,
)
from local_rag_backend.core.services.rag_runtime import RagService
from local_rag_backend.core.services.reranking import RerankingRetriever
from local_rag_backend.infrastructure.concurrency.blocking import run_blocking
from local_rag_backend.infrastructure.ingestion.loaders import (
    ChatGPTLoader,
    GeminiLoader,
    detect_json_export_format,
)
from local_rag_backend.infrastructure.llms.ollama_chat import OllamaGenerator
from local_rag_backend.infrastructure.llms.openai_chat import OpenAIGenerator, create_openai_client
from local_rag_backend.infrastructure.observability.diagnostics import (
    get_document_ids,
    get_documents_count,
    get_history_count,
    get_incomplete_mutation_records_count,
    get_retrieval_index_stats,
)
from local_rag_backend.infrastructure.persistence.sql import (
    HistorySqlStorage,
    SqlDocumentStorage,
    base as db_base,
)
from local_rag_backend.infrastructure.persistence.sql.crud import get_history
from local_rag_backend.infrastructure.persistence.sql.models import Document as DbDocument
from local_rag_backend.infrastructure.persistence.vector.manifest import (
    expected_manifest_config_from_settings,
)
from local_rag_backend.infrastructure.persistence.vector.storage import VectorStorage
from local_rag_backend.infrastructure.retrieval.dense_vector import DenseVectorRetriever
from local_rag_backend.infrastructure.retrieval.hybrid import HybridRetriever
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from sqlalchemy.orm import Session

    from local_rag_backend.core.domain.entities import Document as DomainDocument
    from local_rag_backend.core.domain.types import DocId
    from local_rag_backend.core.ports import (
        DocumentRepoPort,
        EmbedderPort,
        GeneratorPort,
        QAHistoryPort,
        RetrieverPort,
    )
    from local_rag_backend.settings import Settings


T = TypeVar("T")


DEFAULT_DENSE_BACKEND_MESSAGE = (
    "Dense/hybrid retrieval requires an embeddings backend. "
    "Either set OPENAI_API_KEY to use OpenAI embeddings, or install the "
    "'dense-st' extra for SentenceTransformers (e.g. `uv sync --extra dense-st`)."
)


@dataclass(frozen=True)
class _SqlDocsReadPort(DocsReadPort):
    db: Session

    def list_docs_page(self, *, limit: int, offset: int) -> tuple[ListedDocument, ...]:
        rows = (
            self.db.query(DbDocument)
            .order_by(DbDocument.doc_id.asc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return tuple(
            ListedDocument(
                id=str(row.doc_id),
                content=str(row.content),
                external_id=row.external_id,
                source_id=row.source_id,
                metadata=row.metadata_,
            )
            for row in rows
        )


@dataclass(frozen=True)
class _DefaultBlockingExecutor(BlockingExecutorPort):
    run_blocking_fn: Callable[..., Awaitable[Any]]

    async def run_blocking(
        self,
        func: Callable[..., T],
        /,
        *args: Any,
        task_type: BlockingTaskType = "default",
        **kwargs: Any,
    ) -> T:
        return cast(
            T,
            await self.run_blocking_fn(func, *args, task_type=task_type, **kwargs),
        )


@dataclass(frozen=True)
class _SqlHistoryReadPort(HistoryReadPort):
    db: Session

    def list_history_entries(self, *, limit: int, offset: int) -> tuple[HistoryEntry, ...]:
        rows = get_history(db=self.db, limit=limit, offset=offset)
        entries: list[HistoryEntry] = []
        for row in rows:
            created_at = getattr(row, "created_at", None)
            created_at_str = (
                created_at.isoformat()
                if created_at is not None and hasattr(created_at, "isoformat")
                else str(created_at or "")
            )
            source_ids_raw = cast("Any", getattr(row, "source_ids", None)) or []
            entries.append(
                HistoryEntry(
                    id=int(row.id),
                    question=str(row.question),
                    answer=str(row.answer),
                    created_at=created_at_str,
                    source_ids=tuple(str(x) for x in source_ids_raw),
                )
            )
        return tuple(entries)


@dataclass(frozen=True)
class _DefaultHealthDiagnosticsPort(HealthDiagnosticsPort):
    engine: Any

    def ping_database(self) -> None:
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    def get_documents_count(self) -> int:
        return int(get_documents_count(self.engine))

    def get_history_count(self) -> int:
        return int(get_history_count(self.engine))

    def get_document_ids(self) -> tuple[str, ...]:
        return tuple(str(x) for x in get_document_ids(self.engine))

    def get_index_ids(self, *, id_map_path: str) -> tuple[str, ...]:
        ids = json.loads(Path(id_map_path).read_text(encoding="utf-8"))
        return tuple(str(x) for x in ids if str(x).strip())

    def get_retrieval_index_stats(
        self,
        *,
        index_path: str,
        id_map_path: str,
        vector_backend: str,
        dim: int | None = None,
        expected_manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return get_retrieval_index_stats(
            index_path=index_path,
            id_map_path=id_map_path,
            vector_backend=vector_backend,
            dim=dim,
            expected_manifest=expected_manifest,
        )

    def get_incomplete_mutation_records_count(self, *, coordination_dir: Path) -> int:
        return int(get_incomplete_mutation_records_count(coordination_dir=coordination_dir))


@dataclass(frozen=True)
class _DefaultDocsImportLoaderPort(DocsImportLoaderPort):
    def load_texts(self, *, raw: bytes) -> ImportDocsLoadResult:
        detection = detect_json_export_format(raw)
        if detection.fmt == "chatgpt_export":
            loader: ChatGPTLoader | GeminiLoader = ChatGPTLoader(raw)
        elif detection.fmt == "gemini_export":
            loader = GeminiLoader(raw)
        else:
            from local_rag_backend.core.use_cases.docs_import import UnsupportedImportFormatError

            raise UnsupportedImportFormatError()

        items = list(loader.load())
        texts = tuple(item.text for item in items if item.text and item.text.strip())
        return ImportDocsLoadResult(format_detected=detection.fmt, texts=texts)


@dataclass(frozen=True)
class _DefaultRagRuntimeFactory(RagRuntimeFactoryPort):
    doc_repo_factory: Callable[[], DocumentRepoPort]
    history_repo_factory: Callable[[], QAHistoryPort]
    build_retriever_from_config: Callable[..., RetrieverPort]
    build_generator_from_config: Callable[[Any], GeneratorPort]
    rag_service_factory: Callable[..., RagService]

    def run_ask_eval(self, *, question: str, cfg: Any) -> dict[str, Any]:
        doc_repo = self.doc_repo_factory()
        docs = doc_repo.get_all_documents()
        retriever = self.build_retriever_from_config(cfg, doc_repo, preloaded_docs=docs)
        generator = self.build_generator_from_config(cfg)
        history_storage = self.history_repo_factory()
        service = self.rag_service_factory(retriever, generator, history_storage)
        return service.ask(question=question, top_k=int(cfg.k))


class _SqlEvalStoragePort(EvalStoragePort):
    def __init__(self) -> None:
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        session_local = sessionmaker(bind=engine, autocommit=False, autoflush=False)

        # Side-effect import: registers SQLAlchemy model metadata with db_base.
        from local_rag_backend.infrastructure.persistence.sql import models as _models  # noqa: F401

        db_base.ensure_sqlite_schema_compatible(engine_to_use=engine)
        self._doc_repo = SqlDocumentStorage(session_factory=session_local)

    def upsert_dataset_docs(
        self,
        *,
        dataset_id: str,
        docs: tuple[EvalDatasetDocInput, ...],
    ) -> tuple[str, ...]:
        items = [
            SqlDocumentStorage.UpsertDoc(
                external_id=d.external_id,
                content=d.content,
                source_id=d.source_id,
                metadata={"dataset_id": dataset_id, **(d.metadata or {})},
            )
            for d in docs
        ]
        results, _changed, _updated = self._doc_repo.upsert_documents_by_external_id(items)
        return tuple(str(r.external_id) for r in results if r.external_id is not None)

    def list_documents(self) -> tuple[Any, ...]:
        return tuple(self._doc_repo.get_all_documents())

    def get_retriever_storage(self) -> Any:
        return self._doc_repo


@dataclass(frozen=True)
class _DefaultEvalRetrieverFactoryPort(EvalRetrieverFactoryPort):
    def build_sparse_retriever(
        self,
        *,
        storage: EvalStoragePort,
        reranker_enabled: bool,
        candidate_k: int,
        strategy: str,
    ) -> EvalRetrieverPort:
        docs = storage.list_documents()
        corpus = [d.content for d in docs]
        doc_ids = [d.id for d in docs]
        doc_repo = storage.get_retriever_storage()
        base = SparseBM25Retriever(
            documents=corpus,
            doc_ids=doc_ids,
            doc_repo=doc_repo,
            preloaded_docs=docs,
        )
        if reranker_enabled:
            return cast(
                "EvalRetrieverPort",
                RerankingRetriever(base, candidate_k=candidate_k, strategy=strategy),
            )
        return cast("EvalRetrieverPort", base)


@dataclass(frozen=True)
class _OpenAICompatibleOpenRouterClient(OpenRouterClientPort):
    client: Any
    default_model: str

    def generate(self, *, request: OpenRouterGenerateRequest) -> OpenRouterGenerateResult:
        response = self.client.chat.completions.create(
            model=(request.model or self.default_model),
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            messages=[
                {"role": "system", "content": request.system_instruction},
                {"role": "user", "content": request.user_content},
            ],
        )

        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise ValueError("malformed response (missing choices)")

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if content is None:
            text = ""
        elif isinstance(content, str):
            text = content
        else:
            raise ValueError("malformed response content")

        usage_raw = getattr(response, "usage", None)
        usage: OpenRouterUsage | None = None
        if usage_raw is not None:
            usage = OpenRouterUsage(
                prompt_tokens=int(getattr(usage_raw, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage_raw, "completion_tokens", 0) or 0),
                total_tokens=int(getattr(usage_raw, "total_tokens", 0) or 0),
            )
        return OpenRouterGenerateResult(text=text, usage=usage)


def build_docs_read_port(*, db: Session) -> DocsReadPort:
    return _SqlDocsReadPort(db=db)


def build_history_read_port(*, db: Session) -> HistoryReadPort:
    return _SqlHistoryReadPort(db=db)


def build_health_diagnostics_port(*, engine: Any) -> HealthDiagnosticsPort:
    return _DefaultHealthDiagnosticsPort(engine=engine)


def build_expected_manifest_config(*, settings_obj: Settings) -> dict[str, Any]:
    return expected_manifest_config_from_settings(settings_obj)


def build_docs_import_loader_port() -> DocsImportLoaderPort:
    return _DefaultDocsImportLoaderPort()


def build_rag_runtime_factory(
    *,
    doc_repo_factory: Callable[[], DocumentRepoPort] = cast(
        "Callable[[], DocumentRepoPort]", SqlDocumentStorage
    ),
    history_repo_factory: Callable[[], QAHistoryPort] = cast(
        "Callable[[], QAHistoryPort]", HistorySqlStorage
    ),
    build_retriever_from_config: Callable[..., RetrieverPort],
    build_generator_from_config: Callable[[Any], GeneratorPort],
    rag_service_factory: Callable[..., RagService] = RagService,
) -> RagRuntimeFactoryPort:
    return _DefaultRagRuntimeFactory(
        doc_repo_factory=doc_repo_factory,
        history_repo_factory=history_repo_factory,
        build_retriever_from_config=build_retriever_from_config,
        build_generator_from_config=build_generator_from_config,
        rag_service_factory=rag_service_factory,
    )


def build_eval_storage_port() -> EvalStoragePort:
    return _SqlEvalStoragePort()


def build_eval_retriever_factory_port() -> EvalRetrieverFactoryPort:
    return _DefaultEvalRetrieverFactoryPort()


def build_blocking_executor(
    *,
    run_blocking_fn: Callable[..., Awaitable[Any]] = run_blocking,
) -> BlockingExecutorPort:
    return _DefaultBlockingExecutor(run_blocking_fn=run_blocking_fn)


def build_openrouter_client_from_settings(
    *,
    settings_obj: Settings,
    create_openai_client_fn: Callable[..., Any] | None = None,
    openai_client_factory: type[Any] | None = None,
) -> OpenRouterClientPort:
    resolved_create_client = create_openai_client_fn or create_openai_client
    resolved_client_factory = openai_client_factory or OpenAI

    headers: dict[str, str] = {}
    if settings_obj.openrouter_site_url is not None:
        headers["HTTP-Referer"] = settings_obj.openrouter_site_url
    if settings_obj.openrouter_app_title is not None:
        headers["X-Title"] = settings_obj.openrouter_app_title

    client = resolved_create_client(
        api_key=settings_obj.openrouter_api_key,
        base_url=settings_obj.openrouter_base_url,
        default_headers=headers or None,
        timeout=settings_obj.openai_request_timeout,
        client_factory=resolved_client_factory,
    )
    return _OpenAICompatibleOpenRouterClient(
        client=client,
        default_model=str(settings_obj.openrouter_model),
    )


def _build_default_openai_embedder() -> EmbedderPort:
    from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder

    return OpenAIEmbedder()


def _build_default_st_embedder(model_name: str) -> EmbedderPort:
    from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
        SentenceTransformerEmbedder,
    )

    return SentenceTransformerEmbedder(model_name=model_name)


def get_available_llm_providers(*, settings_obj: Settings) -> dict[str, str]:
    providers: dict[str, str] = {}
    if settings_obj.openai_api_key:
        providers["openai"] = "configured"
    if settings_obj.ollama_enabled:
        providers["ollama"] = "enabled"
    if getattr(settings_obj, "openrouter_enabled", False) and getattr(
        settings_obj, "openrouter_api_key", None
    ):
        providers["openrouter"] = "configured"
    return providers


def build_dense_embedder_from_settings(
    *,
    settings_obj: Settings,
    openai_embedder_factory: Callable[[], EmbedderPort] | None = None,
    st_embedder_factory: Callable[[str], EmbedderPort] | None = None,
    missing_backend_message: str | None = None,
) -> EmbedderPort:
    resolved_openai_factory = openai_embedder_factory or _build_default_openai_embedder
    resolved_st_factory = st_embedder_factory or _build_default_st_embedder
    backend_message = missing_backend_message or DEFAULT_DENSE_BACKEND_MESSAGE
    if settings_obj.openai_api_key:
        return resolved_openai_factory()
    try:
        return resolved_st_factory(str(settings_obj.st_embedding_model))
    except RuntimeError as e:
        raise EmbeddingsBackendUnavailableError(backend_message) from e


def build_retriever_with_default_embedder_from_settings(
    *,
    settings_obj: Settings,
    retrieval_mode: str,
    doc_repo: DocumentRepoPort,
    openai_embedder_factory: Callable[[], EmbedderPort] | None = None,
    st_embedder_factory: Callable[[str], EmbedderPort] | None = None,
    missing_backend_message: str | None = None,
    preloaded_docs: Sequence[DomainDocument] | None = None,
    hybrid_alpha: float | None = None,
    enable_reranker: bool | None = None,
    reranker_candidate_k: int | None = None,
    reranker_strategy: str | None = None,
    sparse_retriever_factory: Callable[..., RetrieverPort] = SparseBM25Retriever,
    dense_retriever_factory: Callable[..., RetrieverPort] = DenseVectorRetriever,
    hybrid_retriever_factory: Callable[..., RetrieverPort] = HybridRetriever,
    vector_repo_factory: Callable[..., Any] = VectorStorage,
    reranker_factory: Callable[..., RetrieverPort] = RerankingRetriever,
) -> RetrieverPort:
    def _dense_embedder_factory() -> EmbedderPort:
        return build_dense_embedder_from_settings(
            settings_obj=settings_obj,
            openai_embedder_factory=openai_embedder_factory,
            st_embedder_factory=st_embedder_factory,
            missing_backend_message=missing_backend_message,
        )

    return build_retriever_from_settings(
        settings_obj=settings_obj,
        retrieval_mode=retrieval_mode,
        doc_repo=doc_repo,
        dense_embedder_factory=_dense_embedder_factory,
        preloaded_docs=preloaded_docs,
        hybrid_alpha=hybrid_alpha,
        enable_reranker=enable_reranker,
        reranker_candidate_k=reranker_candidate_k,
        reranker_strategy=reranker_strategy,
        sparse_retriever_factory=sparse_retriever_factory,
        dense_retriever_factory=dense_retriever_factory,
        hybrid_retriever_factory=hybrid_retriever_factory,
        vector_repo_factory=vector_repo_factory,
        reranker_factory=reranker_factory,
    )


def resolve_preferred_llm_provider(*, settings_obj: Settings) -> str:
    if settings_obj.ollama_enabled:
        return "ollama"
    if settings_obj.openai_api_key:
        return "openai"
    if getattr(settings_obj, "openrouter_enabled", False) and getattr(
        settings_obj, "openrouter_api_key", None
    ):
        return "openrouter"
    raise RuntimeError(
        "No LLM configured. Set OPENAI_API_KEY, enable OLLAMA_ENABLED, "
        "or set OPENROUTER_ENABLED=true with OPENROUTER_API_KEY."
    )


def build_retriever_from_settings(
    *,
    settings_obj: Settings,
    retrieval_mode: str,
    doc_repo: DocumentRepoPort,
    dense_embedder_factory: Callable[[], EmbedderPort],
    preloaded_docs: Sequence[DomainDocument] | None = None,
    hybrid_alpha: float | None = None,
    enable_reranker: bool | None = None,
    reranker_candidate_k: int | None = None,
    reranker_strategy: str | None = None,
    sparse_retriever_factory: Callable[..., RetrieverPort] = SparseBM25Retriever,
    dense_retriever_factory: Callable[..., RetrieverPort] = DenseVectorRetriever,
    hybrid_retriever_factory: Callable[..., RetrieverPort] = HybridRetriever,
    vector_repo_factory: Callable[..., Any] = VectorStorage,
    reranker_factory: Callable[..., RetrieverPort] = RerankingRetriever,
) -> RetrieverPort:
    mode = str(retrieval_mode)
    if mode not in {"sparse", "dense", "hybrid"}:
        raise ValueError(f"Unsupported retrieval_mode: {mode}")

    retriever: RetrieverPort
    docs_for_sparse: Sequence[DomainDocument] | None = None
    corpus: list[str] | None = None
    doc_ids: list[DocId] | None = None

    if mode in {"sparse", "hybrid"}:
        docs_for_sparse = (
            list(preloaded_docs) if preloaded_docs is not None else doc_repo.get_all_documents()
        )
        corpus = [d.content for d in docs_for_sparse]
        doc_ids = [d.id for d in docs_for_sparse]

    if mode == "sparse":
        if corpus is None or doc_ids is None or docs_for_sparse is None:
            raise RuntimeError(
                f"Internal error: sparse retrieval vars uninitialized for mode '{mode}'"
            )
        retriever = sparse_retriever_factory(
            documents=corpus,
            doc_ids=doc_ids,
            doc_repo=doc_repo,
            preloaded_docs=docs_for_sparse,
        )
    else:
        embedder = dense_embedder_factory()
        vector_repo = vector_repo_factory(
            index_path=settings_obj.index_path,
            id_map_path=settings_obj.id_map_path,
            dim=embedder.dim,
            backend=getattr(settings_obj, "vector_backend", "auto"),
            settings_obj=settings_obj,
        )
        dense_retriever = dense_retriever_factory(
            embedder=embedder,
            vector_repo=vector_repo,
            doc_repo=doc_repo,
        )
        if mode == "dense":
            retriever = dense_retriever
        else:
            if corpus is None or doc_ids is None or docs_for_sparse is None:
                raise RuntimeError(
                    f"Internal error: hybrid retrieval vars uninitialized for mode '{mode}'"
                )
            sparse_retriever = sparse_retriever_factory(
                documents=corpus,
                doc_ids=doc_ids,
                doc_repo=doc_repo,
                preloaded_docs=docs_for_sparse,
            )
            alpha = (
                hybrid_alpha if hybrid_alpha is not None else settings_obj.hybrid_retrieval_alpha
            )
            retriever = hybrid_retriever_factory(
                dense=dense_retriever,
                sparse=sparse_retriever,
                alpha=alpha,
            )

    reranker_enabled = (
        settings_obj.enable_reranker if enable_reranker is None else bool(enable_reranker)
    )
    if reranker_enabled:
        retriever = reranker_factory(
            retriever,
            candidate_k=(
                settings_obj.reranker_candidate_k
                if reranker_candidate_k is None
                else int(reranker_candidate_k)
            ),
            strategy=(
                settings_obj.reranker_strategy
                if reranker_strategy is None
                else str(reranker_strategy)
            ),
        )
    return retriever


def build_generator_from_settings(
    *,
    settings_obj: Settings,
    llm_provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    prompt_template: str | None = None,
    openai_generator_factory: Callable[..., GeneratorPort] = OpenAIGenerator,
    ollama_generator_factory: Callable[..., GeneratorPort] = OllamaGenerator,
    available_providers: Mapping[str, str] | None = None,
) -> GeneratorPort:
    providers = (
        dict(available_providers)
        if available_providers is not None
        else get_available_llm_providers(settings_obj=settings_obj)
    )
    provider = llm_provider or next(iter(providers), None)

    if not provider:
        raise RuntimeError("No LLM provider available.")
    if provider not in providers:
        raise ValueError(f"LLM provider '{provider}' is not available or configured.")

    if provider == "openrouter":
        headers: dict[str, str] = {}
        if settings_obj.openrouter_site_url is not None:
            headers["HTTP-Referer"] = settings_obj.openrouter_site_url
        if settings_obj.openrouter_app_title is not None:
            headers["X-Title"] = settings_obj.openrouter_app_title
        return openai_generator_factory(
            model=(model or settings_obj.openrouter_model),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            prompt_template=prompt_template,
            api_key=settings_obj.openrouter_api_key,
            base_url=settings_obj.openrouter_base_url,
            extra_headers=headers or None,
        )
    if provider == "openai":
        return openai_generator_factory(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            prompt_template=prompt_template,
        )
    if provider == "ollama":
        return ollama_generator_factory(
            model=model,
            temperature=temperature,
            prompt_template=prompt_template,
        )
    raise ValueError(f"Unsupported llm_provider: {provider}")

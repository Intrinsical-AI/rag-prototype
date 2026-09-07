"""
Shared composition helpers for adapter selection.

This module centralizes policy decisions for:
- dense embedder selection
- retriever wiring
- generator provider wiring
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

from openai import OpenAI
from sqlalchemy import text

from local_rag_backend.composition.evaluation import (
    build_eval_retriever_factory_port as build_eval_retriever_factory_port,
    build_eval_storage_port as build_eval_storage_port,
)
from local_rag_backend.core.domain.retrieval import (
    RetrievalFilter,
    RetrievalRequest,
    RetrievalResult,
    document_matches_filters,
    retrieval_result_from_pairs,
)
from local_rag_backend.core.errors import LLMConfigurationError
from local_rag_backend.core.ports import (
    BlockingExecutorPort,
    BlockingTaskType,
    DocsImportLoaderPort,
    DocsReadPort,
    EmbedderPort,
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
    RetrieverPort,
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
from local_rag_backend.infrastructure.persistence.elasticsearch import (
    ElasticHealthDiagnostics,
)
from local_rag_backend.infrastructure.persistence.sql import (
    HistorySqlStorage,
    SqlDocumentStorage,
    base as db_base,
)
from local_rag_backend.infrastructure.persistence.sql.crud import get_history
from local_rag_backend.infrastructure.persistence.vector.manifest import (
    expected_manifest_config_from_settings,
)
from local_rag_backend.infrastructure.persistence.vector.storage import VectorStorage
from local_rag_backend.infrastructure.retrieval.dense_vector import DenseVectorRetriever
from local_rag_backend.infrastructure.retrieval.hybrid import HybridRetriever
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever
from local_rag_backend.infrastructure.search_backends import (
    ElasticLikeSearchRetriever,
    LocalSplitSearchRetriever,
    SolrSearchRetriever,
)
from local_rag_backend.integrations.embeddings._factory import (
    DEFAULT_DENSE_BACKEND_MESSAGE as DEFAULT_DENSE_BACKEND_MESSAGE,
    _settings_cfg_version,
    build_dense_embedder_from_settings,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping, Sequence

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
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _LocalSparseInputs:
    docs: Sequence[DomainDocument]
    corpus: list[str]
    doc_ids: list[DocId]


@dataclass(frozen=True)
class _RepoDocsReadPort(DocsReadPort):
    doc_repo_factory: Callable[[], DocumentRepoPort]

    def query_docs(
        self,
        *,
        limit: int,
        offset: int,
        filters: tuple[RetrievalFilter, ...],
    ) -> tuple[ListedDocument, ...]:
        rows = sorted(self.doc_repo_factory().get_all_documents(), key=lambda row: str(row.id))
        filtered = (
            [row for row in rows if document_matches_filters(row, filters)]
            if filters
            else list(rows)
        )
        page = filtered[offset : offset + limit]
        return tuple(
            ListedDocument(
                id=str(row.id),
                content=str(row.content),
                external_id=row.external_id,
                source_id=row.source_id,
                metadata=dict(row.metadata or {}) if row.metadata is not None else None,
            )
            for row in page
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
    session_factory: Any

    def list_history_entries(self, *, limit: int, offset: int) -> tuple[HistoryEntry, ...]:
        with self.session_factory() as db:
            rows = get_history(db=db, limit=limit, offset=offset)
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
class _StorageHistoryReadPort(HistoryReadPort):
    history_repo_factory: Callable[[], QAHistoryPort]

    def list_history_entries(self, *, limit: int, offset: int) -> tuple[HistoryEntry, ...]:
        repo = self.history_repo_factory()
        list_entries = getattr(repo, "list_entries", None)
        if not callable(list_entries):
            raise RuntimeError("Configured history backend does not support list_entries.")
        rows = list_entries(limit=limit, offset=offset)
        return tuple(
            HistoryEntry(
                id=int(getattr(row, "id", 0) or 0),
                question=str(getattr(row, "question", "") or ""),
                answer=str(getattr(row, "answer", "") or ""),
                created_at=str(getattr(row, "created_at", "") or ""),
                source_ids=tuple(str(x) for x in (getattr(row, "source_ids", ()) or ())),
            )
            for row in rows
        )


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
        return service.ask(
            question=question,
            top_k=int(cfg.k),
            filters=tuple(item.to_domain() for item in (getattr(cfg, "filters", None) or [])),
            dual_candidate_k=(
                int(cfg.dual_candidate_k)
                if getattr(cfg, "dual_candidate_k", None) is not None
                else None
            ),
            retrieval_mode=str(cfg.retrieval_mode),
        )


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


class _ElasticLexicalRetriever(RetrieverPort):
    def __init__(self, *, vector_repo: Any, doc_repo: DocumentRepoPort) -> None:
        self._vector_repo = vector_repo
        self._doc_repo = doc_repo

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        if request.top_k <= 0:
            return RetrievalResult(items=(), mode_used="sparse", backend_used="elastic_lexical")
        lexical_search = getattr(self._vector_repo, "lexical_search", None)
        if not callable(lexical_search):
            raise RuntimeError("Configured vector backend does not support lexical_search.")
        id_score_pairs = lexical_search(request.query, k=request.top_k)
        if not id_score_pairs:
            return RetrievalResult(items=(), mode_used="sparse", backend_used="elastic_lexical")
        doc_ids, _scores = zip(*id_score_pairs, strict=False)
        docs = self._doc_repo.get(list(doc_ids))
        docs_by_id = {doc.id: doc for doc in docs}
        ordered_docs = []
        ordered_scores = []
        for doc_id, score in id_score_pairs:
            doc = docs_by_id.get(doc_id)
            if doc is None:
                continue
            if not document_matches_filters(doc, request.filters):
                continue
            ordered_docs.append(doc)
            ordered_scores.append(score)
            if len(ordered_docs) >= request.top_k:
                break
        return retrieval_result_from_pairs(
            docs=ordered_docs,
            scores=ordered_scores,
            mode_used="sparse",
            backend_used="elastic_lexical",
            stage="sparse",
        )


def build_docs_read_port(
    *,
    settings_obj: Settings,
    doc_repo_factory: Callable[[], DocumentRepoPort],
) -> DocsReadPort:
    _ = settings_obj
    return _RepoDocsReadPort(doc_repo_factory=doc_repo_factory)


def build_history_read_port(
    *,
    settings_obj: Settings,
    history_repo_factory: Callable[[], QAHistoryPort],
) -> HistoryReadPort:
    if settings_obj.persistence_backend == "elasticsearch":
        return _StorageHistoryReadPort(history_repo_factory=history_repo_factory)
    return _SqlHistoryReadPort(session_factory=db_base.SessionLocal)


def build_health_diagnostics_port(
    *,
    settings_obj: Settings,
    engine: Any,
) -> HealthDiagnosticsPort:
    if settings_obj.persistence_backend == "elasticsearch":
        return ElasticHealthDiagnostics(settings_obj=settings_obj)
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
    reranker_factory: Callable[..., Any] = RerankingRetriever,
) -> RetrieverPort:
    """Build a retriever using the embedder backend chosen from current settings."""

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


def _load_local_sparse_inputs(
    *,
    doc_repo: DocumentRepoPort,
    preloaded_docs: Sequence[DomainDocument] | None,
) -> _LocalSparseInputs:
    docs = (
        list(preloaded_docs) if preloaded_docs is not None else list(doc_repo.get_all_documents())
    )
    return _LocalSparseInputs(
        docs=docs,
        corpus=[doc.content for doc in docs],
        doc_ids=[doc.id for doc in docs],
    )


def _validate_retrieval_backend_compatibility(
    *,
    mode: str,
    persistence_backend: str,
    search_backend: str,
) -> None:
    if mode not in {"sparse", "dense", "dual", "hybrid"}:
        raise ValueError(f"Unsupported retrieval_mode: {mode}")
    if (
        persistence_backend == "elasticsearch"
        and mode == "sparse"
        and search_backend != "elasticsearch"
    ):
        raise ValueError(
            "persistence_backend=elasticsearch supports retrieval_mode=sparse only when "
            "search_backend=elasticsearch"
        )
    if search_backend == "solr" and mode in {"dense", "dual"}:
        raise ValueError("search_backend=solr supports only retrieval_mode=sparse in v1")
    if mode == "hybrid" and search_backend not in {"local_split", "elasticsearch"}:
        raise ValueError(
            "retrieval_mode=hybrid is supported only with search_backend=local_split|elasticsearch"
        )
    if (
        mode == "hybrid"
        and search_backend == "elasticsearch"
        and persistence_backend != "elasticsearch"
    ):
        raise ValueError(
            "retrieval_mode=hybrid with search_backend=elasticsearch requires "
            "persistence_backend=elasticsearch"
        )


def _build_vector_repo_from_settings(
    *,
    settings_obj: Settings,
    vector_repo_factory: Callable[..., Any],
    dim: int | None,
) -> Any:
    return vector_repo_factory(
        index_path=settings_obj.index_path,
        id_map_path=settings_obj.id_map_path,
        dim=dim,
        backend=getattr(settings_obj, "vector_backend", "auto"),
        settings_obj=settings_obj,
    )


def _build_local_split_retriever(
    *,
    settings_obj: Settings,
    mode: str,
    doc_repo: DocumentRepoPort,
    dense_embedder_factory: Callable[[], EmbedderPort],
    sparse_inputs: _LocalSparseInputs | None,
    vector_repo_factory: Callable[..., Any],
) -> RetrieverPort:
    embedder: EmbedderPort | None = None
    vector_repo: Any | None = None
    if mode in {"dense", "dual"}:
        embedder = dense_embedder_factory()
    if mode == "dense":
        vector_repo = _build_vector_repo_from_settings(
            settings_obj=settings_obj,
            vector_repo_factory=vector_repo_factory,
            dim=(embedder.dim if embedder is not None else None),
        )
    return LocalSplitSearchRetriever(
        doc_repo=doc_repo,
        embedder=embedder,
        vector_repo=vector_repo,
        preloaded_docs=(sparse_inputs.docs if sparse_inputs is not None else None),
    )


def _build_remote_search_retriever(
    *,
    settings_obj: Settings,
    search_backend: str,
    embedder: EmbedderPort | None,
) -> RetrieverPort:
    if search_backend == "elasticsearch":
        return ElasticLikeSearchRetriever(
            backend_name="elasticsearch",
            base_url=str(settings_obj.es_base_url or ""),
            docs_index=str(settings_obj.es_docs_index),
            content_field=str(settings_obj.es_content_field),
            embedding_field=str(settings_obj.es_embedding_field),
            request_timeout_s=float(settings_obj.es_request_timeout_s),
            verify_tls=bool(settings_obj.es_verify_tls),
            api_key=settings_obj.es_api_key,
            username=settings_obj.es_username,
            password=settings_obj.es_password,
            embedder=embedder,
            dense_candidate_k=int(settings_obj.es_hybrid_vector_k),
        )
    if search_backend == "opensearch":
        return ElasticLikeSearchRetriever(
            backend_name="opensearch",
            base_url=str(settings_obj.os_base_url or ""),
            docs_index=str(settings_obj.os_docs_index),
            content_field=str(settings_obj.os_content_field),
            embedding_field=str(settings_obj.os_embedding_field),
            request_timeout_s=float(settings_obj.os_request_timeout_s),
            verify_tls=bool(settings_obj.os_verify_tls),
            api_key=settings_obj.os_api_key,
            username=settings_obj.os_username,
            password=settings_obj.os_password,
            embedder=embedder,
            dense_candidate_k=int(settings_obj.os_dense_candidate_k),
        )
    if search_backend == "solr":
        return SolrSearchRetriever(
            base_url=str(settings_obj.solr_base_url or ""),
            core=str(settings_obj.solr_core),
            content_field=str(settings_obj.solr_content_field),
            request_timeout_s=float(settings_obj.solr_request_timeout_s),
        )
    raise ValueError(f"Unsupported search_backend: {search_backend}")


def _build_non_hybrid_retriever_from_settings(
    *,
    settings_obj: Settings,
    mode: str,
    search_backend: str,
    doc_repo: DocumentRepoPort,
    dense_embedder_factory: Callable[[], EmbedderPort],
    sparse_inputs: _LocalSparseInputs | None,
    vector_repo_factory: Callable[..., Any],
) -> RetrieverPort:
    if search_backend == "local_split":
        return _build_local_split_retriever(
            settings_obj=settings_obj,
            mode=mode,
            doc_repo=doc_repo,
            dense_embedder_factory=dense_embedder_factory,
            sparse_inputs=sparse_inputs,
            vector_repo_factory=vector_repo_factory,
        )
    embedder = dense_embedder_factory() if mode in {"dense", "dual"} else None
    return _build_remote_search_retriever(
        settings_obj=settings_obj,
        search_backend=search_backend,
        embedder=embedder,
    )


def _build_hybrid_sparse_retriever(
    *,
    persistence_backend: str,
    search_backend: str,
    doc_repo: DocumentRepoPort,
    vector_repo: Any,
    sparse_inputs: _LocalSparseInputs | None,
    sparse_retriever_factory: Callable[..., RetrieverPort],
) -> RetrieverPort:
    if search_backend == "elasticsearch" or persistence_backend == "elasticsearch":
        return _ElasticLexicalRetriever(vector_repo=vector_repo, doc_repo=doc_repo)
    if sparse_inputs is None:
        raise RuntimeError("Internal error: hybrid retrieval vars uninitialized for local_split")
    return sparse_retriever_factory(
        documents=sparse_inputs.corpus,
        doc_ids=sparse_inputs.doc_ids,
        doc_repo=doc_repo,
        preloaded_docs=sparse_inputs.docs,
    )


def _build_hybrid_retriever_from_settings(
    *,
    settings_obj: Settings,
    persistence_backend: str,
    search_backend: str,
    doc_repo: DocumentRepoPort,
    dense_embedder_factory: Callable[[], EmbedderPort],
    sparse_inputs: _LocalSparseInputs | None,
    hybrid_alpha: float | None,
    sparse_retriever_factory: Callable[..., RetrieverPort],
    dense_retriever_factory: Callable[..., RetrieverPort],
    hybrid_retriever_factory: Callable[..., RetrieverPort],
    vector_repo_factory: Callable[..., Any],
) -> RetrieverPort:
    embedder = dense_embedder_factory()
    vector_repo = _build_vector_repo_from_settings(
        settings_obj=settings_obj,
        vector_repo_factory=vector_repo_factory,
        dim=embedder.dim,
    )
    dense_retriever = dense_retriever_factory(
        embedder=embedder,
        vector_repo=vector_repo,
        doc_repo=doc_repo,
    )
    sparse_retriever = _build_hybrid_sparse_retriever(
        persistence_backend=persistence_backend,
        search_backend=search_backend,
        doc_repo=doc_repo,
        vector_repo=vector_repo,
        sparse_inputs=sparse_inputs,
        sparse_retriever_factory=sparse_retriever_factory,
    )
    return hybrid_retriever_factory(
        dense=dense_retriever,
        sparse=sparse_retriever,
        alpha=(hybrid_alpha if hybrid_alpha is not None else settings_obj.hybrid_retrieval_alpha),
    )


def _apply_reranker_from_settings(
    *,
    settings_obj: Settings,
    retriever: RetrieverPort,
    enable_reranker: bool | None,
    reranker_candidate_k: int | None,
    reranker_strategy: str | None,
    reranker_factory: Callable[..., RetrieverPort],
) -> RetrieverPort:
    reranker_enabled = (
        settings_obj.enable_reranker if enable_reranker is None else bool(enable_reranker)
    )
    if not reranker_enabled:
        return retriever
    return reranker_factory(
        retriever,
        candidate_k=(
            settings_obj.reranker_candidate_k
            if reranker_candidate_k is None
            else int(reranker_candidate_k)
        ),
        strategy=(
            settings_obj.reranker_strategy if reranker_strategy is None else str(reranker_strategy)
        ),
    )


def resolve_preferred_llm_provider(*, settings_obj: Settings) -> str:
    """Return the preferred LLM provider according to the configured precedence."""
    provider: str | None = None
    if settings_obj.ollama_enabled:
        provider = "ollama"
    elif settings_obj.openai_api_key:
        provider = "openai"
    elif getattr(settings_obj, "openrouter_enabled", False) and getattr(
        settings_obj, "openrouter_api_key", None
    ):
        provider = "openrouter"

    if provider is None:
        raise LLMConfigurationError(
            "No LLM configured. Set openai_api_key in config.yaml, enable ollama_enabled, "
            "or enable openrouter_enabled with openrouter_api_key."
        )
    logger.info(
        "llm_provider_selected provider=%s mode=%s backend=%s cfg_version=%s",
        provider,
        str(getattr(settings_obj, "retrieval_mode", "unknown")),
        str(getattr(settings_obj, "search_backend", "local_split")),
        _settings_cfg_version(settings_obj),
    )
    return provider


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
    reranker_factory: Callable[..., Any] = RerankingRetriever,
) -> RetrieverPort:
    """Build the retriever selected by runtime settings and call-site overrides."""

    mode = str(retrieval_mode)
    persistence_backend = str(getattr(settings_obj, "persistence_backend", "local_split"))
    search_backend = str(getattr(settings_obj, "search_backend", "local_split"))
    _validate_retrieval_backend_compatibility(
        mode=mode,
        persistence_backend=persistence_backend,
        search_backend=search_backend,
    )

    sparse_inputs: _LocalSparseInputs | None = None
    if mode in {"sparse", "hybrid", "dual"} and search_backend == "local_split":
        sparse_inputs = _load_local_sparse_inputs(
            doc_repo=doc_repo,
            preloaded_docs=preloaded_docs,
        )

    if mode in {"sparse", "dense", "dual"}:
        retriever = _build_non_hybrid_retriever_from_settings(
            settings_obj=settings_obj,
            mode=mode,
            search_backend=search_backend,
            doc_repo=doc_repo,
            dense_embedder_factory=dense_embedder_factory,
            sparse_inputs=sparse_inputs,
            vector_repo_factory=vector_repo_factory,
        )
    else:
        retriever = _build_hybrid_retriever_from_settings(
            settings_obj=settings_obj,
            persistence_backend=persistence_backend,
            search_backend=search_backend,
            doc_repo=doc_repo,
            dense_embedder_factory=dense_embedder_factory,
            sparse_inputs=sparse_inputs,
            hybrid_alpha=hybrid_alpha,
            sparse_retriever_factory=sparse_retriever_factory,
            dense_retriever_factory=dense_retriever_factory,
            hybrid_retriever_factory=hybrid_retriever_factory,
            vector_repo_factory=vector_repo_factory,
        )

    retriever = _apply_reranker_from_settings(
        settings_obj=settings_obj,
        retriever=retriever,
        enable_reranker=enable_reranker,
        reranker_candidate_k=reranker_candidate_k,
        reranker_strategy=reranker_strategy,
        reranker_factory=reranker_factory,
    )
    logger.info(
        "retriever_selected mode=%s backend=%s cache_key=%s cfg_version=%s",
        mode,
        search_backend,
        (
            str(getattr(settings_obj, "embedding_cache_db_path", "none"))
            if mode in {"dense", "dual", "hybrid"}
            else "none"
        ),
        _settings_cfg_version(settings_obj),
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

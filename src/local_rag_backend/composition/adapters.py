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
from functools import partial
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
)
from local_rag_backend.core.errors import LLMConfigurationError
from local_rag_backend.core.ports import (
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
from local_rag_backend.infrastructure.ingestion.loaders import (
    ChatGPTLoader,
    GeminiLoader,
    detect_json_export_format,
)
from local_rag_backend.infrastructure.llms.ollama_chat import OllamaGenerator
from local_rag_backend.infrastructure.llms.openai_chat import (
    OpenAIGenerator,
    create_openai_client,
    translate_openai_error,
)
from local_rag_backend.infrastructure.observability.diagnostics import (
    get_document_ids,
    get_documents_count,
    get_history_count,
    get_incomplete_mutation_records_count,
    get_retrieval_index_stats,
)
from local_rag_backend.infrastructure.persistence.shared.mutation_journal import FileMutationJournal
from local_rag_backend.infrastructure.persistence.sql.crud import get_history
from local_rag_backend.infrastructure.persistence.vector.manifest import (
    expected_manifest_config_from_settings,
)
from local_rag_backend.infrastructure.persistence.vector.storage import VectorStorage
from local_rag_backend.infrastructure.retrieval.dense_vector import DenseVectorRetriever
from local_rag_backend.infrastructure.retrieval.hybrid import HybridRetriever
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever
from local_rag_backend.infrastructure.search_backends import LocalSplitSearchRetriever
from local_rag_backend.integrations.embeddings._factory import (
    DEFAULT_DENSE_BACKEND_MESSAGE as DEFAULT_DENSE_BACKEND_MESSAGE,
    build_dense_embedder_from_settings,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from local_rag_backend.core.domain.entities import Document as DomainDocument
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
class _RepoDocsReadPort(DocsReadPort):
    doc_repo_factory: Callable[[], DocumentRepoPort]

    def query_docs(
        self,
        *,
        limit: int,
        offset: int,
        filters: tuple[RetrievalFilter, ...],
    ) -> tuple[ListedDocument, ...]:
        page = self.doc_repo_factory().query_documents(limit=limit, offset=offset, filters=filters)
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
class _DefaultHealthDiagnosticsPort(HealthDiagnosticsPort):
    engine: Any
    journal: FileMutationJournal | None = None

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
        if (
            self.journal is not None
            and self.journal.root.resolve()
            == (Path(coordination_dir) / ".mutation_journal").resolve()
        ):
            return self.journal.count_incomplete()
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
        docs = (
            doc_repo.get_all_documents()
            if str(cfg.retrieval_mode) in {"sparse", "dual", "hybrid"}
            else None
        )
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
        try:
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
        except Exception as exc:
            raise translate_openai_error(exc, provider_name="OpenRouter") from exc

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


def build_docs_read_port(
    *,
    settings_obj: Settings,
    doc_repo_factory: Callable[[], DocumentRepoPort],
) -> DocsReadPort:
    _ = settings_obj
    return _RepoDocsReadPort(doc_repo_factory=doc_repo_factory)


def build_history_read_port(
    *,
    session_factory: Any,
) -> HistoryReadPort:
    return _SqlHistoryReadPort(session_factory=session_factory)


def build_health_diagnostics_port(
    *,
    engine: Any,
    journal: FileMutationJournal | None = None,
) -> HealthDiagnosticsPort:
    return _DefaultHealthDiagnosticsPort(engine=engine, journal=journal)


def build_expected_manifest_config(*, settings_obj: Settings) -> dict[str, Any]:
    return expected_manifest_config_from_settings(settings_obj)


def build_docs_import_loader_port() -> DocsImportLoaderPort:
    return _DefaultDocsImportLoaderPort()


def build_rag_runtime_factory(
    *,
    doc_repo_factory: Callable[[], DocumentRepoPort],
    history_repo_factory: Callable[[], QAHistoryPort],
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


def _openrouter_headers(settings_obj: Settings) -> dict[str, str]:
    headers: dict[str, str] = {}
    if settings_obj.openrouter_site_url is not None:
        headers["HTTP-Referer"] = settings_obj.openrouter_site_url
    if settings_obj.openrouter_app_title is not None:
        headers["X-Title"] = settings_obj.openrouter_app_title
    return headers


def build_openrouter_client_from_settings(
    *,
    settings_obj: Settings,
    create_openai_client_fn: Callable[..., Any] | None = None,
    openai_client_factory: type[Any] | None = None,
) -> OpenRouterClientPort:
    resolved_create_client = create_openai_client_fn or create_openai_client
    resolved_client_factory = openai_client_factory or OpenAI

    client = resolved_create_client(
        api_key=settings_obj.openrouter_api_key,
        base_url=settings_obj.openrouter_base_url,
        default_headers=_openrouter_headers(settings_obj) or None,
        timeout=settings_obj.openai_request_timeout,
        client_factory=resolved_client_factory,
    )
    return _OpenAICompatibleOpenRouterClient(
        client=client,
        default_model=str(settings_obj.openrouter_model),
    )


def get_available_llm_providers(*, settings_obj: Settings) -> dict[str, str]:
    providers: dict[str, str] = {}
    if settings_obj.ollama_enabled:
        providers["ollama"] = "enabled"
    if settings_obj.openai_api_key:
        providers["openai"] = "configured"
    if settings_obj.openrouter_configured:
        providers["openrouter"] = "configured"
    return providers


def _preferred_available_llm_provider(providers: Mapping[str, str]) -> str | None:
    return next(
        (name for name in ("ollama", "openai", "openrouter") if name in providers),
        None,
    )


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
        dense_retriever_factory=dense_retriever_factory,
        hybrid_retriever_factory=hybrid_retriever_factory,
        vector_repo_factory=vector_repo_factory,
        reranker_factory=reranker_factory,
    )


def _load_local_sparse_inputs(
    *,
    doc_repo: DocumentRepoPort,
    preloaded_docs: Sequence[DomainDocument] | None,
) -> list[DomainDocument]:
    return (
        list(preloaded_docs) if preloaded_docs is not None else list(doc_repo.get_all_documents())
    )


def _build_cached_sparse_retriever(
    *,
    doc_repo: DocumentRepoPort,
    sparse_inputs: Sequence[DomainDocument] | None,
) -> SparseBM25Retriever | None:
    if sparse_inputs is None:
        return None
    return SparseBM25Retriever(
        documents=[doc.content for doc in sparse_inputs],
        doc_ids=[doc.id for doc in sparse_inputs],
        doc_repo=doc_repo,
        preloaded_docs=sparse_inputs,
    )


def _build_vector_repo_from_settings(
    *,
    settings_obj: Settings,
    vector_repo_factory: Callable[..., Any],
    embedder: EmbedderPort,
) -> Any:
    return vector_repo_factory(
        index_path=settings_obj.index_path,
        id_map_path=settings_obj.id_map_path,
        dim=embedder.dim,
        embedding_identity=embedder.identity,
        backend=getattr(settings_obj, "vector_backend", "auto"),
        settings_obj=settings_obj,
    )


def _build_local_split_retriever(
    *,
    settings_obj: Settings,
    mode: str,
    doc_repo: DocumentRepoPort,
    dense_embedder_factory: Callable[[], EmbedderPort],
    sparse_inputs: Sequence[DomainDocument] | None,
    vector_repo_factory: Callable[..., Any],
) -> RetrieverPort:
    embedder: EmbedderPort | None = None
    vector_repo: Any | None = None
    if mode in {"dense", "dual"}:
        embedder = dense_embedder_factory()
    if mode == "dense":
        assert embedder is not None
        vector_repo = _build_vector_repo_from_settings(
            settings_obj=settings_obj,
            vector_repo_factory=vector_repo_factory,
            embedder=embedder,
        )
    return LocalSplitSearchRetriever(
        doc_repo=doc_repo,
        embedder=embedder,
        vector_repo=vector_repo,
        preloaded_docs=sparse_inputs,
        cached_sparse_retriever=_build_cached_sparse_retriever(
            doc_repo=doc_repo, sparse_inputs=sparse_inputs
        ),
    )


def _build_hybrid_retriever_from_settings(
    *,
    settings_obj: Settings,
    doc_repo: DocumentRepoPort,
    dense_embedder_factory: Callable[[], EmbedderPort],
    sparse_inputs: Sequence[DomainDocument] | None,
    hybrid_alpha: float | None,
    dense_retriever_factory: Callable[..., RetrieverPort],
    hybrid_retriever_factory: Callable[..., RetrieverPort],
    vector_repo_factory: Callable[..., Any],
) -> RetrieverPort:
    embedder = dense_embedder_factory()
    vector_repo = _build_vector_repo_from_settings(
        settings_obj=settings_obj,
        vector_repo_factory=vector_repo_factory,
        embedder=embedder,
    )
    dense_retriever = dense_retriever_factory(
        embedder=embedder,
        vector_repo=vector_repo,
        doc_repo=doc_repo,
    )
    sparse_retriever = LocalSplitSearchRetriever(
        doc_repo=doc_repo,
        preloaded_docs=sparse_inputs,
        cached_sparse_retriever=_build_cached_sparse_retriever(
            doc_repo=doc_repo, sparse_inputs=sparse_inputs
        ),
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
    provider = _preferred_available_llm_provider(
        get_available_llm_providers(settings_obj=settings_obj)
    )

    if provider is None:
        raise LLMConfigurationError(
            "No LLM configured. Set openai_api_key in config.yaml, enable ollama_enabled, "
            "or enable openrouter_enabled with openrouter_api_key."
        )
    logger.info(
        "llm_provider_selected provider=%s mode=%s",
        provider,
        str(getattr(settings_obj, "retrieval_mode", "unknown")),
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
    dense_retriever_factory: Callable[..., RetrieverPort] = DenseVectorRetriever,
    hybrid_retriever_factory: Callable[..., RetrieverPort] = HybridRetriever,
    vector_repo_factory: Callable[..., Any] = VectorStorage,
    reranker_factory: Callable[..., Any] = RerankingRetriever,
) -> RetrieverPort:
    """Build the retriever selected by runtime settings and call-site overrides."""

    mode = str(retrieval_mode)
    if mode not in {"sparse", "dense", "dual", "hybrid"}:
        raise ValueError(f"Unsupported retrieval_mode: {mode}")

    sparse_inputs: list[DomainDocument] | None = None
    if mode in {"sparse", "hybrid", "dual"}:
        sparse_inputs = _load_local_sparse_inputs(
            doc_repo=doc_repo,
            preloaded_docs=preloaded_docs,
        )

    if mode in {"sparse", "dense", "dual"}:
        retriever = _build_local_split_retriever(
            settings_obj=settings_obj,
            mode=mode,
            doc_repo=doc_repo,
            dense_embedder_factory=dense_embedder_factory,
            sparse_inputs=sparse_inputs,
            vector_repo_factory=vector_repo_factory,
        )
    else:
        retriever = _build_hybrid_retriever_from_settings(
            settings_obj=settings_obj,
            doc_repo=doc_repo,
            dense_embedder_factory=dense_embedder_factory,
            sparse_inputs=sparse_inputs,
            hybrid_alpha=hybrid_alpha,
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
        "retriever_selected mode=%s cache_key=%s",
        mode,
        (
            str(getattr(settings_obj, "embedding_cache_db_path", "none"))
            if mode in {"dense", "dual", "hybrid"}
            else "none"
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
    openai_generator_factory: Callable[..., GeneratorPort] | None = None,
    ollama_generator_factory: Callable[..., GeneratorPort] | None = None,
    available_providers: Mapping[str, str] | None = None,
) -> GeneratorPort:
    openai_generator_factory = openai_generator_factory or partial(
        OpenAIGenerator, settings_obj=settings_obj
    )
    ollama_generator_factory = ollama_generator_factory or partial(
        OllamaGenerator, settings_obj=settings_obj
    )
    providers = (
        dict(available_providers)
        if available_providers is not None
        else get_available_llm_providers(settings_obj=settings_obj)
    )
    provider = llm_provider or _preferred_available_llm_provider(providers)

    if not provider:
        raise LLMConfigurationError("No LLM provider available.")
    if provider not in providers:
        raise ValueError(f"LLM provider '{provider}' is not available or configured.")

    if provider == "openrouter":
        return openai_generator_factory(
            model=(model or settings_obj.openrouter_model),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            prompt_template=prompt_template,
            api_key=settings_obj.openrouter_api_key,
            base_url=settings_obj.openrouter_base_url,
            extra_headers=_openrouter_headers(settings_obj) or None,
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

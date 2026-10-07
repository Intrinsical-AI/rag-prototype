"""Application container: single composition point for app/runtime wiring."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from local_rag_backend.composition.adapters import (
    DEFAULT_DENSE_BACKEND_MESSAGE,
    build_dense_embedder_from_settings,
    build_docs_import_loader_port,
    build_docs_read_port,
    build_eval_retriever_factory_port,
    build_eval_storage_port,
    build_expected_manifest_config,
    build_generator_from_settings,
    build_health_diagnostics_port,
    build_history_read_port,
    build_openrouter_client_from_settings,
    build_rag_runtime_factory,
    build_retriever_with_default_embedder_from_settings,
    get_available_llm_providers as get_available_llm_providers_from_settings,
    resolve_preferred_llm_provider,
)
from local_rag_backend.core.ports.contracts import DocsMutationPorts, IndexMutationPorts
from local_rag_backend.core.services.maintenance import (
    rebuild_index_from_db,
)
from local_rag_backend.core.services.prompting import PromptTemplateError, validate_prompt_template
from local_rag_backend.core.services.rag_runtime import RagService
from local_rag_backend.core.services.reranking import RerankingRetriever
from local_rag_backend.core.use_cases.docs_mutation import MutationCoordinator
from local_rag_backend.core.use_cases.index import rebuild_index_sync
from local_rag_backend.infrastructure.concurrency.blocking import BlockingExecutor
from local_rag_backend.infrastructure.concurrency.locks.write_lock import multi_store_write_lock
from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder
from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
    SentenceTransformerEmbedder,
)
from local_rag_backend.infrastructure.llms.ollama_chat import OllamaGenerator
from local_rag_backend.infrastructure.llms.openai_chat import OpenAIGenerator
from local_rag_backend.infrastructure.observability.perf import configure_perf_metrics
from local_rag_backend.infrastructure.persistence.shared.mutation_journal import FileMutationJournal
from local_rag_backend.infrastructure.persistence.sql import (
    HistorySqlStorage,
    SqlDocumentStorage,
    SystemStateStorage,
    base as db_base,
)
from local_rag_backend.infrastructure.persistence.vector.manifest import purge_index_artifacts
from local_rag_backend.infrastructure.persistence.vector.storage import VectorStorage
from local_rag_backend.infrastructure.retrieval.dense_vector import DenseVectorRetriever
from local_rag_backend.infrastructure.retrieval.hybrid import HybridRetriever

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from local_rag_backend.core.domain.entities import Document as DomainDocument
    from local_rag_backend.core.ports import (
        BlockingExecutorPort,
        DocsImportLoaderPort,
        DocsReadPort,
        DocumentRepoPort,
        EmbedderPort,
        EvalRetrieverFactoryPort,
        EvalStoragePort,
        GeneratorPort,
        HealthDiagnosticsPort,
        HistoryReadPort,
        OpenRouterClientPort,
        QAHistoryPort,
        RagRuntimeFactoryPort,
        RetrieverPort,
        VectorRepoPort,
    )
    from local_rag_backend.core.ports.contracts import DocsMutationPorts, IndexMutationPorts
    from local_rag_backend.core.use_cases.rag_query import AskEvalConfigLike
    from local_rag_backend.settings import Settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocsMutationBundle:
    ports: DocsMutationPorts
    import_loader: DocsImportLoaderPort


@dataclass(frozen=True)
class HealthReadinessBundle:
    diagnostics: HealthDiagnosticsPort
    expected_manifest: dict[str, Any]


@dataclass(frozen=True)
class EvalExecutionBundle:
    eval_storage_port: EvalStoragePort
    eval_retriever_factory_port: EvalRetrieverFactoryPort
    reranker_candidate_k: int
    reranker_strategy: str


class AppContainer:
    """Own settings, persistence resources and injectable runtime factories."""

    RAG_SERVICE_STATE_KEY = "rag_service"

    def __init__(
        self,
        *,
        settings_obj: Settings,
        session_factory: sessionmaker[Session] | None = None,
        openai_embedder_factory: Callable[[], EmbedderPort] | None = None,
        st_embedder_factory: Callable[[str], EmbedderPort] | None = None,
        openai_generator_factory: Callable[..., GeneratorPort] | None = None,
        ollama_generator_factory: Callable[..., GeneratorPort] | None = None,
        doc_repo_factory: Callable[[], DocumentRepoPort] | None = None,
        build_upsert_doc: Any | None = None,
        history_repo_factory: Callable[[], QAHistoryPort] | None = None,
        dense_retriever_factory: Callable[..., RetrieverPort] | None = None,
        hybrid_retriever_factory: Callable[..., RetrieverPort] | None = None,
        vector_repo_factory: Callable[..., VectorRepoPort] | None = None,
        reranker_factory: Callable[..., RetrieverPort] | None = None,
        rebuild_fn: Callable[..., int] | None = None,
        purge_index_artifacts_fn: Callable[..., None] | None = None,
        write_lock: Callable[..., Any] | None = None,
        mutation_journal_factory: Callable[..., Any] | None = None,
        mutation_uow_factory: Callable[..., Any] | None = None,
        rag_service_factory: Callable[..., RagService] | None = None,
        system_state_factory: Callable[[], SystemStateStorage] | None = None,
    ) -> None:
        self.settings_obj = settings_obj
        self._blocking_executor = BlockingExecutor(settings_obj=settings_obj)
        self.engine = None
        self.session_factory = session_factory
        self._owns_engine = session_factory is None
        if self.session_factory is None:
            engine_options: dict[str, Any] = {"connect_args": {"check_same_thread": False}}
            if settings_obj.sqlite_url in {"sqlite://", "sqlite:///:memory:"}:
                engine_options.update(poolclass=QueuePool, pool_size=1, max_overflow=0)
            self.engine = create_engine(settings_obj.sqlite_url, **engine_options)
            self.session_factory = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        else:
            self.engine = self.session_factory.kw["bind"]
        default_doc_factory = partial(SqlDocumentStorage, session_factory=self.session_factory)
        default_history_factory = partial(HistorySqlStorage, session_factory=self.session_factory)
        default_state_factory = partial(SystemStateStorage, session_factory=self.session_factory)
        self.openai_embedder_factory = openai_embedder_factory or partial(
            OpenAIEmbedder, settings_obj=settings_obj
        )
        self.st_embedder_factory = st_embedder_factory or (
            lambda model_name: SentenceTransformerEmbedder(
                model_name=model_name, settings_obj=settings_obj
            )
        )
        self.openai_generator_factory = openai_generator_factory or partial(
            OpenAIGenerator, settings_obj=settings_obj
        )
        self.ollama_generator_factory = ollama_generator_factory or partial(
            OllamaGenerator, settings_obj=settings_obj
        )
        self.doc_repo_factory = doc_repo_factory or default_doc_factory
        self.build_upsert_doc = build_upsert_doc or SqlDocumentStorage.UpsertDoc
        self.history_repo_factory = history_repo_factory or default_history_factory
        self.dense_retriever_factory = dense_retriever_factory or DenseVectorRetriever
        self.hybrid_retriever_factory = hybrid_retriever_factory or HybridRetriever
        self.vector_repo_factory = vector_repo_factory or VectorStorage
        self.reranker_factory = reranker_factory or RerankingRetriever
        self.rebuild_fn = rebuild_fn or rebuild_index_from_db
        self.purge_index_artifacts_fn = purge_index_artifacts_fn or purge_index_artifacts
        self.write_lock = write_lock or partial(
            multi_store_write_lock,
            coordination_dir=settings_obj.get_coordination_dir(),
            timeout_s=settings_obj.write_lock_timeout_s,
            poll_s=settings_obj.write_lock_poll_s,
            metrics_path=settings_obj.lock_metrics_path,
        )
        self._mutation_journal = (
            mutation_journal_factory
            or (
                lambda: FileMutationJournal(
                    settings_obj.get_coordination_dir() / ".mutation_journal"
                )
            )
        )()
        self.mutation_journal_factory = lambda: self._mutation_journal
        self.mutation_uow_factory = mutation_uow_factory or (
            partial(db_base.session_uow, session_factory=self.session_factory)
        )
        self.rag_service_factory = rag_service_factory or RagService
        self._system_state = (system_state_factory or default_state_factory)()
        self._rag_service_cache_lock = Lock()
        self._rag_service_cache: RagService | None = None
        self._rag_service_cache_version: int | None = None

    @classmethod
    def from_settings(cls, settings_obj: Settings, **overrides: Any) -> AppContainer:
        return cls(settings_obj=settings_obj, **overrides)

    def initialize(self) -> None:
        """Create resources explicitly at the entrypoint, never on module import."""
        configure_perf_metrics(self.settings_obj.perf_metrics_out_path)
        Path(self.settings_obj.data_dir).mkdir(parents=True, exist_ok=True)
        if self.engine is not None:
            database = self.engine.url.database
            if database and database != ":memory:":
                Path(database).parent.mkdir(parents=True, exist_ok=True)
            db_base.ensure_sqlite_schema_current(engine_to_use=self.engine)

    def close(self) -> None:
        self._blocking_executor.close()
        self.clear_local_rag_service_cache()
        if self.engine is not None and self._owns_engine:
            self.engine.dispose()

    def validate_rag_config(self, config: AskEvalConfigLike) -> list[str]:
        errors = []
        if config.retrieval_mode not in ["sparse", "dense", "dual", "hybrid"]:
            errors.append(f"Invalid retrieval_mode: {config.retrieval_mode}")
        if config.prompt_template is not None:
            try:
                validate_prompt_template(config.prompt_template)
            except PromptTemplateError as e:
                errors.append(f"Invalid prompt_template: {e}")
        return errors

    def get_available_llm_providers(self) -> dict[str, str]:
        return get_available_llm_providers_from_settings(settings_obj=self.settings_obj)

    def build_dense_embedder(
        self,
        *,
        missing_backend_message: str = DEFAULT_DENSE_BACKEND_MESSAGE,
    ) -> EmbedderPort:
        return build_dense_embedder_from_settings(
            settings_obj=self.settings_obj,
            openai_embedder_factory=self.openai_embedder_factory,
            st_embedder_factory=self.st_embedder_factory,
            missing_backend_message=missing_backend_message,
        )

    def run_multi_store_write_locked(self, fn: Callable[[], Any]) -> Any:
        with self.write_lock():
            return fn()

    def build_docs_read_port(self) -> DocsReadPort:
        return build_docs_read_port(
            settings_obj=self.settings_obj,
            doc_repo_factory=self.doc_repo_factory,
        )

    def build_history_read_port(self) -> HistoryReadPort:
        return build_history_read_port(
            session_factory=self.session_factory,
        )

    def build_health_diagnostics_port(self, *, engine: Any | None = None) -> HealthDiagnosticsPort:
        return build_health_diagnostics_port(
            engine=(engine if engine is not None else self.engine),
            journal=self._mutation_journal,
        )

    def build_health_readiness_bundle(self, *, engine: Any | None = None) -> HealthReadinessBundle:
        return HealthReadinessBundle(
            diagnostics=self.build_health_diagnostics_port(engine=engine),
            expected_manifest=self.build_expected_manifest_config(),
        )

    def build_expected_manifest_config(self) -> dict[str, Any]:
        return build_expected_manifest_config(settings_obj=self.settings_obj)

    def build_docs_import_loader_port(self) -> DocsImportLoaderPort:
        return build_docs_import_loader_port()

    def build_docs_mutation_bundle(
        self,
        *,
        missing_backend_message: str = DEFAULT_DENSE_BACKEND_MESSAGE,
        build_embedder: Callable[[], EmbedderPort] | None = None,
    ) -> DocsMutationBundle:
        return DocsMutationBundle(
            ports=self.docs_mutation_ports(
                missing_backend_message=missing_backend_message,
                build_embedder=build_embedder,
            ),
            import_loader=self.build_docs_import_loader_port(),
        )

    def blocking_executor(self) -> BlockingExecutorPort:
        return self._blocking_executor

    def build_openrouter_client(self) -> OpenRouterClientPort:
        return build_openrouter_client_from_settings(settings_obj=self.settings_obj)

    def build_rag_runtime_factory(self) -> RagRuntimeFactoryPort:
        return build_rag_runtime_factory(
            doc_repo_factory=self.doc_repo_factory,
            history_repo_factory=self.history_repo_factory,
            build_retriever_from_config=self.build_retriever_from_config,
            build_generator_from_config=self.build_generator_from_config,
            rag_service_factory=self.rag_service_factory,
        )

    def build_eval_storage_port(self) -> EvalStoragePort:
        return build_eval_storage_port(settings_obj=self.settings_obj)

    def build_eval_retriever_factory_port(self) -> EvalRetrieverFactoryPort:
        return build_eval_retriever_factory_port(
            openai_embedder_factory=self.openai_embedder_factory,
            st_embedder_factory=self.st_embedder_factory,
            dense_retriever_factory=self.dense_retriever_factory,
            hybrid_retriever_factory=self.hybrid_retriever_factory,
            vector_repo_factory=self.vector_repo_factory,
            reranker_factory=self.reranker_factory,
        )

    def build_eval_execution_bundle(self) -> EvalExecutionBundle:
        return EvalExecutionBundle(
            eval_storage_port=self.build_eval_storage_port(),
            eval_retriever_factory_port=self.build_eval_retriever_factory_port(),
            reranker_candidate_k=int(self.settings_obj.reranker_candidate_k),
            reranker_strategy=str(self.settings_obj.reranker_strategy),
        )

    def docs_mutation_ports(
        self,
        *,
        missing_backend_message: str = DEFAULT_DENSE_BACKEND_MESSAGE,
        build_embedder: Callable[[], EmbedderPort] | None = None,
    ) -> DocsMutationPorts:
        resolved_embedder_builder = build_embedder or (
            lambda: self.build_dense_embedder(missing_backend_message=missing_backend_message)
        )
        return DocsMutationPorts(
            build_embedder=resolved_embedder_builder,
            doc_repo_factory=cast("Any", self.doc_repo_factory),
            build_upsert_doc=self.build_upsert_doc,
            vector_repo_factory=self.vector_repo_factory,
            reconcile_index=lambda: rebuild_index_sync(
                settings_obj=self.settings_obj,
                ports=self.index_mutation_ports(build_embedder=resolved_embedder_builder),
            ),
            write_lock=self.write_lock,
            mutation_journal_factory=self.mutation_journal_factory,
            mutation_uow_factory=self.mutation_uow_factory,
            bump_rag_service_version=lambda: self._system_state.bump_version(
                self.RAG_SERVICE_STATE_KEY
            ),
        )

    def index_mutation_ports(
        self,
        *,
        missing_backend_message: str = DEFAULT_DENSE_BACKEND_MESSAGE,
        build_embedder: Callable[[], EmbedderPort] | None = None,
    ) -> IndexMutationPorts:
        resolved_embedder_builder = build_embedder or (
            lambda: self.build_dense_embedder(missing_backend_message=missing_backend_message)
        )
        return IndexMutationPorts(
            build_embedder=resolved_embedder_builder,
            doc_repo_factory=self.doc_repo_factory,
            vector_repo_factory=self.vector_repo_factory,
            purge_index_artifacts_fn=self.purge_index_artifacts_fn,
            rebuild_fn=self.rebuild_fn,
        )

    def recover_incomplete_doc_mutations(self, *, limit: int = 100) -> int:
        coordinator = MutationCoordinator(
            settings_obj=self.settings_obj,
            ports=self.docs_mutation_ports(),
        )
        return coordinator.recover_incomplete(limit=limit)

    def build_retriever_from_config(
        self,
        cfg: AskEvalConfigLike,
        doc_repo: DocumentRepoPort,
        *,
        preloaded_docs: Sequence[DomainDocument] | None = None,
        missing_backend_message: str = DEFAULT_DENSE_BACKEND_MESSAGE,
    ) -> RetrieverPort:
        return build_retriever_with_default_embedder_from_settings(
            settings_obj=self.settings_obj,
            retrieval_mode=cfg.retrieval_mode,
            doc_repo=doc_repo,
            openai_embedder_factory=self.openai_embedder_factory,
            st_embedder_factory=self.st_embedder_factory,
            missing_backend_message=missing_backend_message,
            preloaded_docs=preloaded_docs,
            hybrid_alpha=cfg.hybrid_alpha,
            enable_reranker=self.settings_obj.enable_reranker,
            reranker_candidate_k=self.settings_obj.reranker_candidate_k,
            reranker_strategy=self.settings_obj.reranker_strategy,
            dense_retriever_factory=self.dense_retriever_factory,
            hybrid_retriever_factory=self.hybrid_retriever_factory,
            vector_repo_factory=self.vector_repo_factory,
            reranker_factory=self.reranker_factory,
        )

    def build_generator_from_config(self, cfg: AskEvalConfigLike) -> GeneratorPort:
        available_providers = self.get_available_llm_providers()
        return build_generator_from_settings(
            settings_obj=self.settings_obj,
            llm_provider=cfg.llm_provider,
            model=cfg.model,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            prompt_template=cfg.prompt_template,
            openai_generator_factory=self.openai_generator_factory,
            ollama_generator_factory=self.ollama_generator_factory,
            available_providers=available_providers,
        )

    def build_rag_service(self) -> RagService:
        doc_repo = self.doc_repo_factory()
        retriever = build_retriever_with_default_embedder_from_settings(
            settings_obj=self.settings_obj,
            retrieval_mode=self.settings_obj.retrieval_mode,
            doc_repo=doc_repo,
            openai_embedder_factory=self.openai_embedder_factory,
            st_embedder_factory=self.st_embedder_factory,
            dense_retriever_factory=self.dense_retriever_factory,
            hybrid_retriever_factory=self.hybrid_retriever_factory,
            vector_repo_factory=self.vector_repo_factory,
            reranker_factory=self.reranker_factory,
        )

        preferred_provider = resolve_preferred_llm_provider(settings_obj=self.settings_obj)
        generator = build_generator_from_settings(
            settings_obj=self.settings_obj,
            llm_provider=preferred_provider,
            openai_generator_factory=self.openai_generator_factory,
            ollama_generator_factory=self.ollama_generator_factory,
        )

        history_repo = self.history_repo_factory()
        return self.rag_service_factory(
            retriever=retriever,
            generator=generator,
            history_storage=history_repo,
        )

    def read_rag_service_version(self) -> int:
        try:
            return self._system_state.get_version(self.RAG_SERVICE_STATE_KEY)
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to read RAG service version from system_state: %s", e)
            return 0

    def get_rag_service(self) -> RagService:
        version = self.read_rag_service_version()
        with self._rag_service_cache_lock:
            if (
                self._rag_service_cache is not None
                and self._rag_service_cache_version is not None
                and self._rag_service_cache_version == version
            ):
                return self._rag_service_cache

            service = self.build_rag_service()
            self._rag_service_cache = service
            self._rag_service_cache_version = version
            return service

    def reset_rag_service(self) -> None:
        try:
            self._system_state.bump_version(self.RAG_SERVICE_STATE_KEY)
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to bump RAG service version in system_state: %s", e)
        self.clear_local_rag_service_cache()

    def clear_local_rag_service_cache(self) -> None:
        with self._rag_service_cache_lock:
            self._rag_service_cache = None
            self._rag_service_cache_version = None


__all__ = ["AppContainer"]

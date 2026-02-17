"""
Shared application wiring helpers for API routers.

This module centralizes runtime composition and adapter wiring so transport routers
can stay thin and focused on HTTP concerns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import HTTPException

from local_rag_backend.app.composition import (
    build_dense_embedder_from_settings,
    build_generator_from_settings,
    build_retriever_with_default_embedder_from_settings,
    get_available_llm_providers as get_available_llm_providers_from_settings,
)
from local_rag_backend.app.services.mutation_ports import (
    build_docs_mutation_ports,
    build_index_mutation_ports,
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
from local_rag_backend.core.services.reranking import RerankingRetriever
from local_rag_backend.core.services.write_lock import multi_store_write_lock
from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder
from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
    SentenceTransformerEmbedder,
)
from local_rag_backend.infrastructure.llms.ollama_chat import OllamaGenerator
from local_rag_backend.infrastructure.llms.openai_chat import OpenAIGenerator
from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
from local_rag_backend.infrastructure.persistence.faiss.manifest import purge_index_artifacts
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage
from local_rag_backend.infrastructure.retrieval.dense_faiss import DenseFaissRetriever
from local_rag_backend.infrastructure.retrieval.hybrid import HybridRetriever
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever
from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from local_rag_backend.app.schemas import AskEvalConfig
    from local_rag_backend.app.services.ports import DocsMutationPorts, IndexMutationPorts
    from local_rag_backend.core.domain.entities import Document as DomainDocument
    from local_rag_backend.core.ports import (
        DocumentRepoPort,
        EmbedderPort,
        GeneratorPort,
        RetrieverPort,
    )

_DENSE_BACKEND_ERROR_MESSAGE = (
    "Dense/hybrid operations require an embeddings backend. "
    "Set OPENAI_API_KEY or install the 'dense-st' extra."
)


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


def build_embedder_for_dense() -> EmbedderPort:
    """Build a dense embedder for runtime operations and map backend errors to 400."""
    try:
        return build_dense_embedder_from_settings(
            settings_obj=settings,
            openai_embedder_factory=OpenAIEmbedder,
            st_embedder_factory=lambda model_name: SentenceTransformerEmbedder(model_name=model_name),
            missing_backend_message=_DENSE_BACKEND_ERROR_MESSAGE,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=_DENSE_BACKEND_ERROR_MESSAGE) from e


def run_multi_store_write_locked(fn: Callable[[], Any]) -> Any:
    """Execute a mutating callable under the cross-process write lock."""
    with multi_store_write_lock():
        return fn()


def docs_mutation_ports() -> DocsMutationPorts:
    """Build docs mutation ports wired to current settings/infrastructure adapters."""
    return build_docs_mutation_ports(
        build_embedder=build_embedder_for_dense,
        doc_repo_factory=cast("Any", lambda: SqlDocumentStorage()),
        build_upsert_doc=SqlDocumentStorage.UpsertDoc,
        vector_repo_factory=FaissVectorStorage,
        precompute_vectors_fn=precompute_vectors_for_changed_items,
        sync_dense_fn=sync_dense_after_upsert,
        rebuild_fn=rebuild_index_from_db,
        delete_docs_fn=delete_documents_multi_store,
        delete_external_ids_fn=delete_external_ids_multi_store,
    )


def index_mutation_ports() -> IndexMutationPorts:
    """Build index mutation ports wired to current settings/infrastructure adapters."""
    return build_index_mutation_ports(
        build_embedder=build_embedder_for_dense,
        doc_repo_factory=lambda: SqlDocumentStorage(),
        vector_repo_factory=FaissVectorStorage,
        purge_index_artifacts_fn=purge_index_artifacts,
        rebuild_fn=rebuild_index_from_db,
    )


def build_retriever_from_config(
    cfg: AskEvalConfig,
    doc_repo: DocumentRepoPort,
    *,
    preloaded_docs: Sequence[DomainDocument] | None = None,
) -> RetrieverPort:
    """Build a retriever instance based on dynamic per-request configuration."""
    try:
        return build_retriever_with_default_embedder_from_settings(
            settings_obj=settings,
            retrieval_mode=cfg.retrieval_mode,
            doc_repo=doc_repo,
            openai_embedder_factory=OpenAIEmbedder,
            st_embedder_factory=lambda model_name: SentenceTransformerEmbedder(model_name=model_name),
            missing_backend_message=_DENSE_BACKEND_ERROR_MESSAGE,
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


def build_generator_from_config(cfg: AskEvalConfig) -> GeneratorPort:
    """Build a generator instance based on dynamic per-request configuration."""
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

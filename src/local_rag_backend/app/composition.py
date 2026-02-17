"""
Shared composition helpers for adapter selection.

This module centralizes policy decisions for:
- dense embedder selection
- retriever wiring
- generator provider wiring
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_backend.core.errors import EmbeddingsBackendUnavailableError
from local_rag_backend.core.services.reranking import RerankingRetriever
from local_rag_backend.infrastructure.llms.ollama_chat import OllamaGenerator
from local_rag_backend.infrastructure.llms.openai_chat import OpenAIGenerator
from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
from local_rag_backend.infrastructure.retrieval.dense_faiss import DenseFaissRetriever
from local_rag_backend.infrastructure.retrieval.hybrid import HybridRetriever
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any

    from local_rag_backend.core.domain.entities import Document as DomainDocument
    from local_rag_backend.core.ports import (
        DocumentRepoPort,
        EmbedderPort,
        GeneratorPort,
        RetrieverPort,
    )
    from local_rag_backend.settings import Settings


DEFAULT_DENSE_BACKEND_MESSAGE = (
    "Dense/hybrid retrieval requires an embeddings backend. "
    "Either set OPENAI_API_KEY to use OpenAI embeddings, or install the "
    "'dense-st' extra for SentenceTransformers (e.g. `uv sync --extra dense-st`)."
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
    dense_retriever_factory: Callable[..., RetrieverPort] = DenseFaissRetriever,
    hybrid_retriever_factory: Callable[..., RetrieverPort] = HybridRetriever,
    vector_repo_factory: Callable[..., Any] = FaissVectorStorage,
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
    raise RuntimeError("No LLM configured. Set OPENAI_API_KEY or enable OLLAMA_ENABLED.")


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
    dense_retriever_factory: Callable[..., RetrieverPort] = DenseFaissRetriever,
    hybrid_retriever_factory: Callable[..., RetrieverPort] = HybridRetriever,
    vector_repo_factory: Callable[..., Any] = FaissVectorStorage,
    reranker_factory: Callable[..., RetrieverPort] = RerankingRetriever,
) -> RetrieverPort:
    mode = str(retrieval_mode)
    if mode not in {"sparse", "dense", "hybrid"}:
        raise ValueError(f"Unsupported retrieval_mode: {mode}")

    retriever: RetrieverPort
    docs_for_sparse: Sequence[DomainDocument] | None = None
    corpus: list[str] | None = None
    doc_ids: list[int] | None = None

    if mode in {"sparse", "hybrid"}:
        docs_for_sparse = list(preloaded_docs) if preloaded_docs is not None else doc_repo.get_all_documents()
        corpus = [d.content for d in docs_for_sparse]
        doc_ids = [d.id for d in docs_for_sparse]

    if mode == "sparse":
        assert corpus is not None
        assert doc_ids is not None
        assert docs_for_sparse is not None
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
        )
        dense_retriever = dense_retriever_factory(
            embedder=embedder,
            faiss_index=vector_repo,
            doc_repo=doc_repo,
        )
        if mode == "dense":
            retriever = dense_retriever
        else:
            assert corpus is not None
            assert doc_ids is not None
            assert docs_for_sparse is not None
            sparse_retriever = sparse_retriever_factory(
                documents=corpus,
                doc_ids=doc_ids,
                doc_repo=doc_repo,
                preloaded_docs=docs_for_sparse,
            )
            alpha = hybrid_alpha if hybrid_alpha is not None else settings_obj.hybrid_retrieval_alpha
            retriever = hybrid_retriever_factory(
                dense=dense_retriever,
                sparse=sparse_retriever,
                alpha=alpha,
            )

    reranker_enabled = settings_obj.enable_reranker if enable_reranker is None else bool(enable_reranker)
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

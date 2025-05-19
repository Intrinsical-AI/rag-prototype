# src/app/factories.py

"""
Singleton lifecycle for RagService:
- Initialized ONCE on first call to get_rag_service().
- To reset (e.g. for tests or settings reload), call reset_rag_service().
- Safe for most FastAPI single-process dev runs, but NOT truly process-safe.
- For multiprocess (e.g., uvicorn workers>1), each process holds its own singleton.
"""

import logging

from src.core.services.rag import RagService
from src.infrastructure.embeddings.sentence_transformers import (
    SentenceTransformerEmbedder,
)
from src.infrastructure.llms.ollama_chat import OllamaGenerator
from src.infrastructure.llms.openai_chat import OpenAIGenerator
from src.infrastructure.persistence.faiss.index import FaissIndex
from src.infrastructure.persistence.sqlalchemy.sql_ import (
    HistorySqlStorage,
    SqlDocumentStorage,
)
from src.infrastructure.retrieval.dense_faiss import DenseFaissRetriever
from src.infrastructure.retrieval.hybrid import HybridRetriever
from src.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever
from src.settings import settings
from src.utils import get_corpus_and_ids

logger = logging.getLogger(__name__)


def check_faiss_sql_consistency(doc_ids, faiss_index):
    # validación exhaustiva - puede ser costoso si la base de datos crece mucho
    sql_set = set(doc_ids)
    faiss_set = set(faiss_index.id_map)
    if sql_set != faiss_set:
        logger.warning(
            f"FAISS/SQL id mismatch. SQL: {sql_set - faiss_set}, FAISS: {faiss_set - sql_set}."
        )
    if len(doc_ids) != len(faiss_index.id_map):
        logger.warning(
            f"FAISS id_map ({len(faiss_index.id_map)}) and SQL docs ({len(doc_ids)}) count mismatch. Possible index desync."
        )


def get_generator():
    if settings.ollama_enabled:
        logger.info(f"Using OllamaGenerator (model: {settings.ollama_model})")
        return OllamaGenerator()
    elif settings.openai_api_key:
        logger.info(f"Using OpenAIGenerator (model: {settings.openai_model})")
        return OpenAIGenerator()
    else:
        logger.error("No LLM generator configured")
        raise RuntimeError("No LLM generator configured")


def get_retriever():
    doc_repo = SqlDocumentStorage()
    corpus, doc_ids = get_corpus_and_ids(doc_repo)

    if settings.retrieval_mode == "dense":
        embedder = SentenceTransformerEmbedder(model_name=settings.st_embedding_model)
        faiss_index = FaissIndex(
            index_path=settings.index_path,
            id_map_path=settings.id_map_path,
            dim=embedder.dim,
        )
        check_faiss_sql_consistency(doc_ids, faiss_index)
        logger.info(f"Using DenseFaissRetriever (docs: {len(doc_ids)})")
        return DenseFaissRetriever(
            embedder=embedder, faiss_index=faiss_index, doc_repo=doc_repo
        )
    elif settings.retrieval_mode == "sparse":
        logger.info(f"Using SparseBM25Retriever (docs: {len(doc_ids)})")
        return SparseBM25Retriever(documents=corpus, doc_ids=doc_ids, doc_repo=doc_repo)
    elif settings.retrieval_mode == "hybrid":
        embedder = SentenceTransformerEmbedder(model_name=settings.st_embedding_model)
        faiss_index = FaissIndex(
            index_path=settings.index_path,
            id_map_path=settings.id_map_path,
            dim=embedder.dim,
        )
        check_faiss_sql_consistency(doc_ids, faiss_index)
        dense = DenseFaissRetriever(
            embedder=embedder, faiss_index=faiss_index, doc_repo=doc_repo
        )
        sparse = SparseBM25Retriever(
            documents=corpus, doc_ids=doc_ids, doc_repo=doc_repo
        )
        logger.info(f"Using HybridRetriever (dense+bm25) (docs: {len(doc_ids)})")
        return HybridRetriever(dense=dense, sparse=sparse, alpha=0.5)
    else:
        logger.error(f"Unsupported retrieval_mode: {settings.retrieval_mode}")
        raise ValueError(f"Unsupported retrieval_mode: {settings.retrieval_mode}")


_rag_service = None


def get_rag_service(force_reload: bool = False) -> RagService:
    global _rag_service
    if force_reload or _rag_service is None:
        retriever = get_retriever()
        generator = get_generator()
        history_storage = HistorySqlStorage()
        _rag_service = RagService(retriever, generator, history_storage)
    return _rag_service


def reset_rag_service():
    """
    Reset the singleton RAG service (for tests, dev, or controlled reload).
    """
    global _rag_service
    _rag_service = None

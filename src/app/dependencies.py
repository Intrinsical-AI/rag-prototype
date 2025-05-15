from __future__ import annotations

import logging
import sys
import csv
from pathlib import Path

import requests
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from src.settings import settings
from src.core.rag import RagService
from src.utils import preprocess_text
from src.adapters.retrieval.sparse_bm25 import SparseBM25Retriever
from src.adapters.retrieval.dense_faiss import DenseFaissRetriever
from src.adapters.embeddings.sentence_transformers import SentenceTransformerEmbedder
from src.adapters.generation.openai_chat import OpenAIGenerator
from src.adapters.generation.ollama_chat import OllamaGenerator

import src.db.base as db_base
from src.db.models import Base as AppDeclarativeBase, Document as DbDocument
from src.db.crud import add_documents as crud_add_documents


# --------------------------------------------------------------------------- #
# Logging Configuration
# --------------------------------------------------------------------------- #
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
logger = logging.getLogger(__name__)

_rag_service: RagService | None = None


# --------------------------------------------------------------------------- #
# LLM Generator Selection
# --------------------------------------------------------------------------- #
def _choose_generator():
    """Select and return an appropriate LLM generator based on configuration."""
    ollama_tags_url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"

    if settings.ollama_enabled:
        try:
            if requests.get(ollama_tags_url, timeout=2).status_code == 200:
                logger.info("Using OllamaGenerator (primary).")
                return OllamaGenerator()
        except requests.RequestException:
            logger.warning("Primary Ollama health-check failed.")

    if settings.openai_api_key:
        logger.info("Using OpenAIGenerator.")
        return OpenAIGenerator()

    if settings.ollama_enabled:  # Fallback
        try:
            if requests.get(ollama_tags_url, timeout=2).status_code == 200:
                logger.info("Using OllamaGenerator (fallback).")
                return OllamaGenerator()
        except requests.RequestException:
            logger.warning("Fallback Ollama health-check failed.")

    raise RuntimeError("LLM Generator could not be initialized: no OpenAI key and Ollama unreachable.")


# --------------------------------------------------------------------------- #
# RagService Initialization
# --------------------------------------------------------------------------- #
def init_rag_service():
    """Initialize the singleton `RagService`."""
    global _rag_service
    logger.info("Initializing RagService...")

    is_in_memory_db = "mode=memory" in settings.sqlite_url or ":memory:" in settings.sqlite_url

    if not db_base.engine or str(db_base.engine.url) != settings.sqlite_url or (
        is_in_memory_db and not isinstance(db_base.engine.pool, StaticPool)
    ):
        logger.warning("Engine mismatch or missing. Reconfiguring engine.")
        pool_kwargs = {"poolclass": StaticPool} if is_in_memory_db else {}
        new_engine = create_engine(
            settings.sqlite_url,
            connect_args={"check_same_thread": False},
            **pool_kwargs,
        )
        db_base.engine = new_engine
        db_base.SessionLocal.configure(bind=new_engine)
        logger.info("Engine reconfigured to %s", new_engine.url)

    logger.info("Ensuring tables exist at %s", db_base.engine.url)
    AppDeclarativeBase.metadata.create_all(bind=db_base.engine)

    with db_base.SessionLocal() as db:
        doc_count = db.query(DbDocument).count()
        faq_csv_path = Path(settings.faq_csv)

        if doc_count == 0 and faq_csv_path.is_file():
            logger.info("Database empty. Populating from CSV: %s", faq_csv_path)
            texts = []
            try:
                with faq_csv_path.open(newline="", encoding="utf-8") as fh:
                    reader = csv.reader(fh, delimiter=";")
                    if settings.csv_has_header:
                        next(reader, None)
                    for i, row in enumerate(reader):
                        if len(row) >= 2:
                            texts.append(preprocess_text(f"{row[0]} {row[1]}"))
                        else:
                            logger.warning("Row %d malformed: %s", i+1, row)
            except Exception as e:
                logger.error("Error reading CSV: %s", e, exc_info=True)

            if texts:
                crud_add_documents(db, texts)
                db.commit()
                logger.info("Committed %d documents to DB.", len(texts))

    with db_base.SessionLocal() as db:
        docs = db.query(DbDocument).all()
        corpus = [doc.content for doc in docs]
        ids = [doc.id for doc in docs]

    if settings.retrieval_mode == "dense":
        index_path = Path(settings.index_path)
        id_map_path = Path(settings.id_map_path)
        if not index_path.is_file() or not id_map_path.is_file():
            logger.warning("Falling back to sparse retrieval: dense artifacts missing.")
            retriever = SparseBM25Retriever(corpus, ids)
        else:
            logger.info("Using DenseFaissRetriever.")
            retriever = DenseFaissRetriever(embedder=SentenceTransformerEmbedder())
    else:
        logger.info("Using SparseBM25Retriever.")
        retriever = SparseBM25Retriever(corpus, ids)

    generator = _choose_generator()
    _rag_service = RagService(retriever, generator)
    logger.info("RagService initialized successfully.")


def get_rag_service() -> RagService:
    """FastAPI dependency providing the initialized RagService singleton."""
    assert _rag_service is not None, "RagService has not been initialized."
    return _rag_service

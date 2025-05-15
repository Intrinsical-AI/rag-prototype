from __future__ import annotations

import logging
import sys
from pathlib import Path
import csv

import requests
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool # Removed Pool as it's not directly used here for type hint
# from sqlalchemy.orm import Session as SQLAlchemySession # Not strictly needed as type hint if using db_base.SessionLocal

from src.settings import settings
from src.core.rag import RagService
from src.utils import preprocess_text
from src.adapters.retrieval.sparse_bm25 import SparseBM25Retriever
from src.adapters.retrieval.dense_faiss import DenseFaissRetriever
from src.adapters.embeddings.sentence_transformers import (
    SentenceTransformerEmbedder,
)
from src.adapters.generation.openai_chat import OpenAIGenerator
from src.adapters.generation.ollama_chat import OllamaGenerator

import src.db.base as db_base
from src.db.models import Base as AppDeclarativeBase, Document as DbDocument
from src.db.crud import add_documents as crud_add_documents


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
        stream=sys.stdout,
    )
logger = logging.getLogger(__name__)

_rag_service: RagService | None = None


# --------------------------------------------------------------------------- #
# Generador LLM
# --------------------------------------------------------------------------- #
def _choose_generator():
    # ... (sin cambios)
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

    if settings.ollama_enabled: # Fallback
        try:
            if requests.get(ollama_tags_url, timeout=2).status_code == 200:
                logger.info("Using OllamaGenerator (fallback).")
                return OllamaGenerator()
        except requests.RequestException:
            logger.warning("Fallback Ollama health-check failed.")

    raise RuntimeError(
        "LLM Generator could not be initialized: no OpenAI key and Ollama unreachable."
    )

# --------------------------------------------------------------------------- #
# Inicialización de RagService
# --------------------------------------------------------------------------- #
def init_rag_service():
    """(Re)inicializa el singleton `RagService`."""
    global _rag_service
    logger.info("DEPS: Starting init_rag_service...")
    # _rag_service se resetea a None por la fixture reset_rag_service_singleton en conftest.py
    # antes de llamar a esta función en los tests que lo necesiten.

    is_in_memory_db = "mode=memory" in settings.sqlite_url or ":memory:" in settings.sqlite_url
    logger.info(f"DEPS: Current settings.sqlite_url (from settings module): {settings.sqlite_url}")
    if db_base.engine: # db_base.engine should be patched by conftest.py
        logger.info(f"DEPS: Current db_base.engine.url (from db_base module): {str(db_base.engine.url)} (id: {id(db_base.engine)})")
        logger.info(f"DEPS: db_base.engine.pool is StaticPool: {isinstance(db_base.engine.pool, StaticPool)}")
    else:
        logger.error("DEPS: db_base.engine is None! This should have been patched by conftest.py for tests.")
        # This case should ideally not happen if conftest.py is doing its job.

    # En tests, conftest.py DEBE haber configurado settings.sqlite_url, db_base.engine,
    # y db_base.SessionLocal ANTES de que init_rag_service se llame.
    # Esta comprobación es una salvaguarda. Si se dispara en tests, hay un problema en conftest.py.
    if not db_base.engine or str(db_base.engine.url) != settings.sqlite_url or \
       (is_in_memory_db and not isinstance(db_base.engine.pool, StaticPool)):
        logger.warning(
            f"DEPS: Engine mismatch or incorrect pool for in-memory, OR db_base.engine not set. "
            f"Current Engine URL: {str(db_base.engine.url) if db_base.engine else 'None'}, "
            f"Pool: {type(db_base.engine.pool) if db_base.engine else 'N/A'}, "
            f"Settings URL: {settings.sqlite_url}. Reconfiguring db_base.engine..."
        )
        # This reconfiguration should ideally not be needed if conftest.py is correct.
        pool_kwargs = (
            {"poolclass": StaticPool} if is_in_memory_db else {}
        )
        new_engine = create_engine(
            settings.sqlite_url, # This URL should be the test one, patched by conftest
            connect_args={"check_same_thread": False},
            **pool_kwargs,
        )
        db_base.engine = new_engine # Actualiza el engine en el módulo db_base
        db_base.SessionLocal.configure(bind=new_engine) # Reconfigura SessionLocal en db_base
        logger.info(f"DEPS: Reconfigured db_base.engine to {new_engine.url} (id: {id(new_engine)}) with pool {type(new_engine.pool)}")
    else:
        logger.info("DEPS: db_base.engine and settings.sqlite_url are synchronized. Using pre-configured engine from db_base.")

    # Crear tablas. Es idempotente.
    # Si conftest.py ya creó las tablas en db_base.engine, esto no hace nada.
    # Si la rama de "Reconfiguring" de arriba se ejecutó, crea tablas en el new_engine.
    logger.info("DEPS: Ensuring tables in %s …", db_base.engine.url)
    AppDeclarativeBase.metadata.create_all(bind=db_base.engine)

    # --- Poblar BD desde CSV si está vacía ----------------------------------
    # db_base.SessionLocal() usará el engine configurado (idealmente por conftest.py)
    with db_base.SessionLocal() as db: # type: ignore # SQLAlchemySession or SQLModel Session
        engine_for_session = db.get_bind()
        logger.info(f"DEPS: Session for DB populating is using engine {id(engine_for_session)} for URL {engine_for_session.url}")

        doc_count = db.query(DbDocument).count()
        logger.info(f"DEPS: Documents currently in DB: {doc_count}")
        faq_csv_path = Path(settings.faq_csv) # settings.faq_csv debe ser parcheado por conftest.py para tests
        logger.info(f"DEPS: Checking FAQ CSV at: {faq_csv_path}, exists: {faq_csv_path.is_file()}")

        if doc_count == 0 and faq_csv_path.is_file():
            logger.info("DEPS: DB empty and FAQ CSV found; populating from %s …", faq_csv_path)
            texts: list[str] = []
            try:
                with faq_csv_path.open(newline="", encoding="utf-8") as fh:
                    reader = csv.reader(fh, delimiter=";")
                    if settings.csv_has_header:
                        try:
                            next(reader, None)
                            logger.info("DEPS: Skipped CSV header.")
                        except StopIteration:
                            logger.warning("DEPS: CSV header expected but file was empty after header.")
                    for i, row in enumerate(reader):
                        if len(row) >= 2:
                            texts.append(preprocess_text(f"{row[0]} {row[1]}"))
                        else:
                            logger.warning(f"DEPS: Row {i+1} in CSV has fewer than 2 columns: {row}")
            except Exception as e:
                logger.error(f"DEPS: Error reading CSV {faq_csv_path}: {e}", exc_info=True)

            if texts:
                logger.info(f"DEPS: Read {len(texts)} texts from CSV. Adding to DB...")
                crud_add_documents(db, texts)
                db.commit()
                logger.info("DEPS: Committed %d documents from CSV.", len(texts))
                logger.info(f"DEPS: Documents in DB after commit: {db.query(DbDocument).count()}")
            else:
                logger.info("DEPS: No texts extracted from CSV to insert.")
        elif doc_count > 0:
            logger.info("DEPS: DB not empty, skipping CSV population.")
        elif not faq_csv_path.is_file():
            logger.info("DEPS: FAQ CSV file not found at %s, skipping CSV population.", faq_csv_path)

    # --- Recuperador --------------------------------------------------------
    corpus: list[str] = []
    ids: list[int] = [] # type: ignore[assignment]
    with db_base.SessionLocal() as db: # type: ignore
        docs = db.query(DbDocument).all()
        corpus = [d.content for d in docs]
        ids = [d.id for d in docs] # type: ignore
        logger.info(f"DEPS: Loaded {len(corpus)} documents for retriever setup.")

    if not corpus:
        logger.warning("DEPS: Corpus is empty. Retriever might not function as expected.")

    retriever: SparseBM25Retriever | DenseFaissRetriever
    if settings.retrieval_mode == "dense":
        # ... (lógica dense sin cambios)
        index_path = Path(settings.index_path)
        id_map_path = Path(settings.id_map_path)
        if not index_path.is_file() or not id_map_path.is_file():
            logger.warning(
                "Falling back to sparse retrieval: dense artifacts missing (index: %s, map: %s).",
                index_path.exists(), id_map_path.exists()
            )
            if not corpus:
                 logger.error("DEPS: Corpus is empty, cannot initialize SparseBM25Retriever. Service will be unusable.")
            retriever = SparseBM25Retriever(corpus, ids)
        else:
            logger.info("DEPS: Using DenseFaissRetriever.")
            retriever = DenseFaissRetriever(
                embedder=SentenceTransformerEmbedder()
            )
    else:
        logger.info("DEPS: Using SparseBM25Retriever.")
        if not corpus and ids:
            logger.warning("DEPS: Corpus empty but IDs exist. Retriever might be misconfigured.")
        elif not corpus and not ids:
            logger.info("DEPS: Corpus and IDs are empty for SparseBM25Retriever.")
        retriever = SparseBM25Retriever(corpus, ids)

    # --- Generador ----------------------------------------------------------
    generator = _choose_generator()
    _rag_service = RagService(retriever, generator)
    logger.info(f"DEPS: RagService (id: {id(_rag_service)}) initialized successfully with {settings.retrieval_mode} retriever.")


def get_rag_service() -> RagService:
    """
    Dependencia FastAPI que entrega el singleton RagService ya creado.
    Lanza AssertionError si init_rag_service() no ha sido llamada previamente.
    """
    # El test test_get_rag_service_raises_assertion_error_if_not_initialized
    # espera que esta aserción falle si _rag_service es None.
    # La fixture reset_rag_service_singleton (en conftest.py) pone _rag_service a None.
    assert _rag_service is not None, "RagService not initialised" # Mensaje coincide con el test
    return _rag_service
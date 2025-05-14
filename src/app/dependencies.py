"""
src/app/dependencies.py

Gestiona el singleton `RagService` para la aplicación FastAPI:

* Sincroniza el engine de SQLAlchemy con `settings.sqlite_url`.
* Crea las tablas si no existen.
* Pobla la BD con el CSV de FAQ cuando está vacía.
* Selecciona automáticamente el generador (OpenAI u Ollama) y
  el recuperador (BM25 o FAISS) según la configuración.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import requests
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

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

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Singleton
_rag_service: RagService | None = None


# --------------------------------------------------------------------------- #
# Generador LLM
# --------------------------------------------------------------------------- #
def _choose_generator():
    """Devuelve la instancia apropiada de generador."""
    ollama_tags_url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"

    # — Ollama habilitado y responde —
    if settings.ollama_enabled:
        try:
            if requests.get(ollama_tags_url, timeout=2).status_code == 200:
                logger.info("Using OllamaGenerator (primary).")
                return OllamaGenerator()
        except requests.RequestException:
            logger.warning("Primary Ollama health-check failed.")

    # — OpenAI disponible —
    if settings.openai_api_key:
        logger.info("Using OpenAIGenerator.")
        return OpenAIGenerator()

    # — Ollama en modo fallback —
    if settings.ollama_enabled:
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

    # Importes diferidos (respetan monkey-patches de los tests)
    import src.db.base as db_base
    from src.db.models import Base, Document as DbDocument
    from src.db.crud import add_documents as crud_add_documents

    # --- Sincronizar engine / SessionLocal ----------------------------------
    if str(db_base.engine.url) != settings.sqlite_url:
        # Para in-memory necesitamos StaticPool para compartir la misma conexión
        pool_kwargs = (
            {"poolclass": StaticPool} if settings.sqlite_url.endswith(":memory:") else {}
        )
        new_engine = create_engine(
            settings.sqlite_url,
            connect_args={"check_same_thread": False},
            **pool_kwargs,
        )
        db_base.engine = new_engine
        db_base.SessionLocal.configure(bind=new_engine)

    # Crear tablas
    logger.info("Ensuring tables in %s …", db_base.engine.url)
    Base.metadata.create_all(bind=db_base.engine)

    # --- Poblar BD desde CSV si está vacía ----------------------------------
    with db_base.SessionLocal() as db:
        if db.query(DbDocument).count() == 0 and Path(settings.faq_csv).is_file():
            logger.info("DB empty; populating from %s …", settings.faq_csv)
            import csv

            texts: list[str] = []
            with open(settings.faq_csv, newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh, delimiter=";")
                if settings.csv_has_header:
                    next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        texts.append(preprocess_text(f"{row[0]} {row[1]}"))

            if texts:
                crud_add_documents(db, texts)
                logger.info("Inserted %d documents from CSV.", len(texts))

    # --- Recuperador --------------------------------------------------------
    with db_base.SessionLocal() as db:
        docs = db.query(DbDocument).all()
        corpus = [d.content for d in docs]
        ids = [d.id for d in docs]

    if settings.retrieval_mode == "dense":
        if not Path(settings.index_path).is_file() or not Path(
            settings.id_map_path
        ).is_file():
            logger.warning(
                "Falling back to sparse retrieval: dense artifacts missing."
            )
            retriever = SparseBM25Retriever(corpus, ids)
        else:
            retriever = DenseFaissRetriever(
                embedder=SentenceTransformerEmbedder()
            )
    else:
        retriever = SparseBM25Retriever(corpus, ids)

    # --- Generador ----------------------------------------------------------
    generator = _choose_generator()
    _rag_service = RagService(retriever, generator)


def get_rag_service() -> RagService:
    """Dependencia FastAPI que entrega el singleton ya creado."""
    assert _rag_service is not None, "RagService not initialised"
    return _rag_service

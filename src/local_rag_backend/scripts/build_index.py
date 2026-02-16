# scripts/build_index.py

"""
Script to initialize the database from the FAQ CSV and build necessary indexes.
It uses a dedicated database engine and session for this process.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder
from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
    SentenceTransformerEmbedder,
)

# Import models to ensure they are registered with Base.metadata
# To ensure table creation
from local_rag_backend.infrastructure.persistence.sqlalchemy import (
    base as db_base,
    models as _models,
)
from local_rag_backend.infrastructure.persistence.sqlalchemy.base import Base as AppDeclarativeBase
from local_rag_backend.settings import settings

logger = logging.getLogger(__name__)
_ = _models
T = TypeVar("T")
# Configure logging to see script and data_loader output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from local_rag_backend.core.ports import EmbedderPort


def _run_with_multi_store_write_lock(operation: Callable[[], T]) -> T:
    from local_rag_backend.core.services.write_lock import multi_store_write_lock

    with multi_store_write_lock(coordination_dir=settings.get_coordination_dir()):
        return operation()


def main() -> None:
    """
    Main entry point for the build_index script.
    Initializes DB and optionally FAISS index using functionality from data_loader.
    """
    logger.info("Starting build_index script...")
    # Ensure data dir exists before touching SQLite.
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create a dedicated engine and SessionLocal for this script
    #    This avoids conflicts with the main app engine.
    logger.info(f"Using database URL: {settings.sqlite_url}")
    is_in_memory = "mode=memory" in settings.sqlite_url or ":memory:" in settings.sqlite_url
    pool_kwargs = {"poolclass": StaticPool} if is_in_memory else {}
    script_engine = create_engine(
        settings.sqlite_url,
        connect_args={"check_same_thread": False},  # Necesario para SQLite si se usa en threads
        **pool_kwargs,
    )
    script_session_local = sessionmaker(bind=script_engine, autocommit=False, autoflush=False)

    # 2. Ensure the database schema exists
    logger.info(f"Ensuring database schema exists at {script_engine.url}...")
    try:
        AppDeclarativeBase.metadata.create_all(bind=script_engine)
        db_base.ensure_sqlite_documents_autoincrement(
            engine_to_use=script_engine, id_map_path=str(settings.id_map_path)
        )
        db_base.ensure_sqlite_documents_identity_columns(engine_to_use=script_engine)
        logger.info("Database schema ensured (tables created if they didn't exist).")
    except Exception as e:
        logger.error(f"Failed to ensure database schema: {e}", exc_info=True)
        raise RuntimeError("Unable to ensure SQLite schema before build-index.") from e

    # 3. Embedder (for dense or hybrid mode)
    embedder_for_indexing: EmbedderPort | None = None
    if settings.retrieval_mode in ["dense", "hybrid"]:
        logger.info(
            f"{settings.retrieval_mode.title()} retrieval mode detected. Initializing embedder for indexing."
        )
        if settings.openai_api_key:
            embedder_for_indexing = OpenAIEmbedder()
        else:
            embedder_for_indexing = SentenceTransformerEmbedder(
                model_name=settings.st_embedding_model
            )

    # 4. Use ETL logic directly (similar to bootstrap.py)
    try:
        import csv

        from local_rag_backend.core.services.etl import ETLService
        from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
        from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

        # Read CSV (explicit -> repo fallback)
        csv_path = Path(settings.faq_csv)
        if not csv_path.is_file():
            repo_csv = Path(__file__).resolve().parents[3] / "data" / "faq.csv"
            if repo_csv.is_file():
                csv_path = repo_csv
        texts = []
        if csv_path.is_file():
            with csv_path.open(encoding="utf-8") as fh:
                reader = csv.reader(fh, delimiter=";")
                if settings.csv_has_header:
                    next(reader, None)
                for i, row in enumerate(reader, 1):
                    if len(row) < 2:
                        logger.warning(f"Row {i} skipped (len={len(row)}): {row}")
                        continue
                    texts.append(f"{row[0].strip()} {row[1].strip()}")
        else:
            raise FileNotFoundError(
                f"CSV file not found at {csv_path}. Set FAQ_CSV to a valid path."
            )

        if not texts:
            raise ValueError("No texts found in CSV.")

        logger.info(f"Parsed {len(texts)} documents from CSV.")

        # ETL Service
        doc_repo = SqlDocumentStorage(session_factory=script_session_local)

        if settings.retrieval_mode in ["dense", "hybrid"] and embedder_for_indexing:
            vector_repo = FaissVectorStorage(
                index_path=settings.index_path,
                id_map_path=settings.id_map_path,
                dim=embedder_for_indexing.dim,
            )
            etl = ETLService(doc_repo, vector_repo, embedder_for_indexing)
            ids = list(_run_with_multi_store_write_lock(lambda: etl.ingest(texts)))
            logger.info(f"Ingested {len(ids)} docs into SQL and FAISS.")
        else:
            # SQL only for sparse mode
            ids = list(_run_with_multi_store_write_lock(lambda: doc_repo.store_documents(texts)))
            logger.info(f"Ingested {len(ids)} docs into SQL only (sparse mode).")

        logger.info("build_index script finished successfully.")

    except FileNotFoundError as e:
        logger.error(f"Halting script: {e}")
        raise
    except ValueError as e:  # For example, dimension mismatch of embedder
        logger.error(f"Halting script due to value error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during build_index: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

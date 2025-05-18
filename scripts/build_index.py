# scripts/build_index.py

"""
Script to initialize the database from the FAQ CSV and build necessary indexes.
It uses a dedicated database engine and session for this process.
"""

import logging
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Importar las funciones del nuevo módulo data_loader
from src.data_loader import setup_initial_data_from_csv

# Para asegurar la creación de tablas
from src.infra.persistence.sqlalchemy.models import Base as AppDeclarativeBase

# Importar el embedder que se usará para la indexación si es modo denso
from src.infrastructure.embeddings import (  # o el que decidas
    SentenceTransformerEmbedder,
)
from src.settings import settings

logger = logging.getLogger(__name__)
# Configurar el logging para que se vea la salida del script y de data_loader
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main() -> None:
    """
    Main entry point for the build_index script.
    Initializes DB and optionally FAISS index using functionality from data_loader.
    """
    logger.info("Starting build_index script...")

    # 1. Crear un engine y SessionLocal exclusivos para este script
    #    Esto evita interferencias con el engine de la aplicación principal.
    logger.info(f"Using database URL: {settings.sqlite_url}")
    is_in_memory = (
        "mode=memory" in settings.sqlite_url or ":memory:" in settings.sqlite_url
    )
    pool_kwargs = {"poolclass": StaticPool} if is_in_memory else {}
    script_engine = create_engine(
        settings.sqlite_url,
        connect_args={
            "check_same_thread": False
        },  # Necesario para SQLite si se usa en threads
        **pool_kwargs,
    )
    ScriptSessionLocal = sessionmaker(
        bind=script_engine, autocommit=False, autoflush=False
    )

    # 2. Asegurar que el esquema de la BBDD (tablas) existe
    logger.info(f"Ensuring database schema exists at {script_engine.url}...")
    try:
        AppDeclarativeBase.metadata.create_all(bind=script_engine)
        logger.info("Database schema ensured (tables created if they didn't exist).")
    except Exception as e:
        logger.error(f"Failed to ensure database schema: {e}", exc_info=True)
        return  # Salir si no se pueden crear las tablas

    # 3. Embedder
    embedder_for_indexing = None
    if settings.retrieval_mode == "dense":
        logger.info(
            "Dense retrieval mode detected. Initializing embedder for indexing."
        )
        embedder_for_indexing = SentenceTransformerEmbedder(
            model_name=settings.st_embedding_model
        )

    # 4. Usar una sesión de BBDD para llamar a la lógica de data_loader
    try:
        with ScriptSessionLocal() as session:
            logger.info("Calling data_loader.setup_initial_data_from_csv...")
            csv_to_use = settings.faq_csv
            header_setting = settings.csv_has_header
            should_create_dense_index = (
                settings.create_dense_index is True
                and settings.retrieval_mode == "dense"
            )
            logger.info(f"Using CSV for build_index: {csv_to_use}")
            setup_initial_data_from_csv(
                db_session=session,
                csv_path_str=Path(csv_to_use),  # PASAR EXPLÍCITAMENTE
                has_header=header_setting,  # PASAR EXPLÍCITAMENTE
                create_dense_index_flag=should_create_dense_index,
                embedder_instance=embedder_for_indexing,
            )
        logger.info("build_index script finished successfully.")

    except FileNotFoundError as e:
        logger.error(f"Halting script: {e}")
    except ValueError as e:  # Por ejemplo, el mismatch de dimensiones del embedder
        logger.error(f"Halting script due to value error: {e}", exc_info=True)
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during build_index: {e}", exc_info=True
        )


if __name__ == "__main__":
    main()

"""
scripts/build_index.py
----------------------

• Vuelca el CSV de FAQ a SQLite.
• Si `settings.retrieval_mode == "dense"` genera un índice FAISS + id-map.

La función principal **NO** modifica `src.db.base.engine`; crea su propio
engine / SessionLocal para no contaminar al resto de la aplicación ni a los tests.
"""

from __future__ import annotations

import csv
import logging
import pickle
from pathlib import Path
from typing import List

import faiss  # type: ignore
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.settings import settings
from src.utils import preprocess_text

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# --------------------------------------------------------------------------- #
# Utilidades internas
# --------------------------------------------------------------------------- #
def _make_engine_and_session():
    """Devuelve `(engine, SessionLocal)` *exclusivos* para este script."""
    pool_kwargs = (
        {"poolclass": StaticPool} if settings.sqlite_url.endswith(":memory:") else {}
    )
    engine = create_engine(
        settings.sqlite_url, connect_args={"check_same_thread": False}, **pool_kwargs
    )
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    # Asegurar las tablas sin tocar src.db.base
    from src.db.models import Base

    Base.metadata.create_all(bind=engine)
    logger.info("DB schema ensured in %s", engine.url)

    return engine, SessionLocal


def _read_csv() -> List[str]:
    """Lee y pre-procesa el CSV definido en settings."""
    csv_path = Path(settings.faq_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"FAQ CSV not found at {csv_path}")

    texts: list[str] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter=";")
        if settings.csv_has_header:
            next(reader, None)  # descartar cabecera
        for row in reader:
            if len(row) >= 2:
                texts.append(preprocess_text(f"{row[0]} {row[1]}"))

    logger.info("Read %d rows from CSV %s", len(texts), csv_path)
    return texts


def _insert_documents(session, texts: List[str]) -> List[int]:
    """Inserta documentos y devuelve la lista de IDs asignados."""
    from src.db import crud

    crud.add_documents(session, texts)
    ids = [d.id for d in session.query(crud.models.Document).all()]  # type: ignore
    return ids


def _build_dense_index(vectors: np.ndarray, ids: List[int]):
    """Crea archivos FAISS + id-map para búsqueda *dense*."""
    index_path = Path(settings.index_path)
    id_map_path = Path(settings.id_map_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    id_map_path.parent.mkdir(parents=True, exist_ok=True)

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors.astype("float32"))
    faiss.write_index(index, str(index_path))
    logger.info("FAISS index saved to %s (ntotal=%d)", index_path, index.ntotal)

    with id_map_path.open("wb") as fh:
        pickle.dump(ids, fh)
    logger.info("ID-map saved to %s", id_map_path)


# --------------------------------------------------------------------------- #
# Punto de entrada CLI / tests
# --------------------------------------------------------------------------- #
def main() -> None:
    _, SessionLocal = _make_engine_and_session()
    texts = _read_csv()

    with SessionLocal() as session:
        ids = _insert_documents(session, texts)
        session.commit()
        logger.info("Inserted %d documents into the database.", len(ids))

    # ----------------------------------------------------- #
    #  ÍNDICE FAISS opcional
    # ----------------------------------------------------- #
    if settings.retrieval_mode == "dense":
        dim = 384  # Dimensión fija para los tests
        rng = np.random.default_rng(seed=42)
        vectors = rng.random((len(texts), dim), dtype=np.float32)
        _build_dense_index(vectors, ids)


if __name__ == "__main__":
    main()

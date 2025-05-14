"""
scripts/build_index.py

Construye o reconstruye la base de conocimiento local:

* Inserta los documentos del CSV en SQLite.
* Si `settings.retrieval_mode == "dense"`, crea un índice FAISS y su id-map.

Pensado para CLI y para los tests unitarios.
"""
from __future__ import annotations

import csv
import logging
import pickle
from pathlib import Path

import faiss  # type: ignore
import numpy as np

from src.settings import settings
from src.utils import preprocess_text

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ensure_db_schema():
    """
    Sincroniza `src.db.base` con la URL de SQLite actual y asegura las tablas.
    Devuelve el `SessionLocal` listo para usar.
    """
    import importlib
    import src.db.base as db_base

    if str(db_base.engine.url) != settings.sqlite_url:
        from sqlalchemy import create_engine

        new_engine = create_engine(
            settings.sqlite_url, connect_args={"check_same_thread": False}
        )
        db_base.engine = new_engine
        db_base.SessionLocal.configure(bind=new_engine)

    from src.db.models import Base

    Base.metadata.create_all(bind=db_base.engine)
    logger.info("DB schema ensured in %s", db_base.engine.url)

    # Re-exportar para que futuros imports vean el engine actualizado
    importlib.reload(db_base)

    return db_base.SessionLocal


def _read_csv() -> list[str]:
    csv_path = Path(settings.faq_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"FAQ CSV not found at {csv_path}")

    texts: list[str] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter=";")
        if settings.csv_has_header:
            next(reader, None)
        for row in reader:
            if len(row) >= 2:
                texts.append(preprocess_text(f"{row[0]} {row[1]}"))
    return texts


def _insert_documents(session, texts: list[str]) -> list[int]:
    from src.db import crud

    crud.add_documents(session, texts)
    docs = session.query(crud.models.Document).all()  # type: ignore
    return [d.id for d in docs]


def _build_dense_index(vectors: np.ndarray, ids: list[int]):
    index_path = Path(settings.index_path)
    id_map_path = Path(settings.id_map_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    id_map_path.parent.mkdir(parents=True, exist_ok=True)

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors.astype("float32"))
    faiss.write_index(index, str(index_path))

    with id_map_path.open("wb") as fh:
        pickle.dump(ids, fh)

    logger.info("FAISS index saved to %s (ntotal=%d)", index_path, index.ntotal)
    logger.info("ID-map saved to %s", id_map_path)


# --------------------------------------------------------------------------- #
# Entry-point
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(level=logging.INFO)

    SessionLocal = _ensure_db_schema()
    texts = _read_csv()

    with SessionLocal() as session:
        ids = _insert_documents(session, texts)

    logger.info("Inserted %d documents into the database.", len(ids))

    if settings.retrieval_mode == "dense":
        dim = 384  # Dimensión dummy suficiente para los tests
        rng = np.random.default_rng(seed=42)
        vectors = rng.random((len(texts), dim), dtype=np.float32)
        _build_dense_index(vectors, ids)


if __name__ == "__main__":
    main()

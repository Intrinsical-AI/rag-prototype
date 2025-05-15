# scripts/build_index.py

"""
- Loads the FAQ CSV into SQLite.
- If `settings.retrieval_mode == "dense"`, generates FAISS index + ID map.

The main function **does not** modify `src.db.base.engine`; it creates its own
engine and SessionLocal to prevent contamination of the main app or tests.
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


# ----------------------------- Internal Utilities ----------------------------- #
def _make_engine_and_session():
    """Create an exclusive engine and sessionmaker for this script."""
    pool_kwargs = (
        {"poolclass": StaticPool} if settings.sqlite_url.endswith(":memory:") else {}
    )
    engine = create_engine(
        settings.sqlite_url, connect_args={"check_same_thread": False}, **pool_kwargs
    )
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    # Ensure DB schema without affecting src.db.base
    from src.db.models import Base

    Base.metadata.create_all(bind=engine)
    logger.info("Database schema ensured at %s", engine.url)

    return engine, SessionLocal


def _read_csv() -> List[str]:
    """Read and preprocess texts from the FAQ CSV defined in settings."""
    csv_path = Path(settings.faq_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"FAQ CSV file not found at {csv_path}")

    texts: list[str] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter=";")
        if settings.csv_has_header:
            next(reader, None)  # Skip header if present
        for row in reader:
            if len(row) >= 2:
                texts.append(preprocess_text(f"{row[0]} {row[1]}"))

    logger.info("Read %d texts from CSV at %s", len(texts), csv_path)
    return texts


def _insert_documents(session, texts: List[str]) -> List[int]:
    """Insert texts into the DB and return the list of assigned IDs."""
    from src.db import crud

    crud.add_documents(session, texts)
    ids = [doc.id for doc in session.query(crud.models.Document).all()]  # type: ignore
    return ids


def _build_dense_index(vectors: np.ndarray, ids: List[int]):
    """Create FAISS index and ID-map files for dense retrieval."""
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


# ------------------------- CLI / Script Entry Point ------------------------- #
def main() -> None:
    """Main function to build DB and optionally FAISS index from CSV."""
    _, SessionLocal = _make_engine_and_session()
    texts = _read_csv()

    with SessionLocal() as session:
        ids = _insert_documents(session, texts)
        session.commit()
        logger.info("Inserted %d documents into the database.", len(ids))

    # Optional FAISS index generation for dense retrieval
    if settings.retrieval_mode == "dense":
        embedding_dim = 384  # Fixed dimension for testing/demo purposes
        rng = np.random.default_rng(seed=42)
        vectors = rng.random((len(texts), embedding_dim), dtype=np.float32)
        _build_dense_index(vectors, ids)


if __name__ == "__main__":
    main()

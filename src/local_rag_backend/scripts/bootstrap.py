# scripts/bootstrap.py

import csv
import sys
from importlib import resources
from pathlib import Path

# IMPORTS PARA BD DINÁMICA
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from local_rag_backend.core.services.etl import ETLService
from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
    SentenceTransformerEmbedder,
)
from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
from local_rag_backend.infrastructure.persistence.sqlalchemy import models  # noqa: F401
from local_rag_backend.infrastructure.persistence.sqlalchemy.base import Base
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage
from local_rag_backend.settings import settings

DELIMITER = ";"


def main():
    # 1) Creamos engine y sesión basados en la URL actualizada
    engine = create_engine(
        settings.sqlite_url, connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    # 2) Leemos CSV (local primero, luego paquete)
    csv_path = Path(settings.faq_csv)
    texts = []
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter=DELIMITER)
            if settings.csv_has_header:
                next(reader, None)
            for i, row in enumerate(reader, 1):
                if len(row) < 2:
                    print(f"[WARN] Row {i} skipped (len={len(row)}): {row}")
                    continue
                texts.append(f"{row[0].strip()} {row[1].strip()}")
    else:
        # Fallback to packaged CSV
        try:
            pkg_csv = resources.files("local_rag_backend.data").joinpath("faq.csv")
            with pkg_csv.open("r", encoding="utf-8") as fh:
                reader = csv.reader(fh, delimiter=DELIMITER)
                if settings.csv_has_header:
                    next(reader, None)
                for i, row in enumerate(reader, 1):
                    if len(row) < 2:
                        print(f"[WARN] Row {i} skipped (len={len(row)}): {row}")
                        continue
                    texts.append(f"{row[0].strip()} {row[1].strip()}")
            print("[INFO] Loaded packaged sample data (local CSV not found).")
        except Exception:
            print(f"[ERR] CSV file not found at {csv_path} and no packaged sample available.")
            sys.exit(1)
    if not texts:
        print("[ERR] No texts found in CSV.")
        sys.exit(1)

    print(f"[INFO] Parsed {len(texts)} documents from CSV.")

    # 3) Invocamos ETL con nuestro SessionLocal freshly-built
    doc_repo = SqlDocumentStorage(session_factory=SessionLocal)

    if settings.retrieval_mode in ["dense", "hybrid"]:
        embedder = SentenceTransformerEmbedder(model_name=settings.st_embedding_model)
        vector_repo = FaissVectorStorage(
            index_path=settings.index_path,
            id_map_path=settings.id_map_path,
            dim=embedder.dim,
        )
        etl = ETLService(doc_repo, vector_repo, embedder)
        ids = etl.ingest(texts)
        print(f"[OK] Ingested {len(ids)} docs into SQL and FAISS.")
    else:
        # Solo SQL para sparse mode
        ids = doc_repo.store_documents(texts)
        print(f"[OK] Ingested {len(ids)} docs into SQL only (sparse mode).")


if __name__ == "__main__":
    main()

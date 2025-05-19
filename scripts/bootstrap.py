# scripts/bootstrap.py

import csv
import sys
from pathlib import Path

# IMPORTS PARA BD DINÁMICA
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.services.etl import ETLService
from src.infrastructure.embeddings.sentence_transformers import (
    SentenceTransformerEmbedder,
)
from src.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
from src.infrastructure.persistence.sqlalchemy.base import Base
from src.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage
from src.settings import settings

DELIMITER = ";"


def main():
    # 1) Creamos engine y sesión basados en la URL actualizada
    engine = create_engine(
        settings.sqlite_url, connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    # 2️⃣  ¡muy importante!  -> que todo el proyecto use ESTA sesión
    from src.infrastructure.persistence.sqlalchemy import base as _db_base

    _db_base.SessionLocal = SessionLocal  # ← sobrescribimos la global

    from src.infrastructure.persistence.sqlalchemy import sql_ as _sql_mod

    # 3️⃣  Que las próximas llamadas a SqlDocumentStorage() sin argumentos
    #     utilicen ESTA nueva SessionLocal -------------------------------
    _sql_mod.SessionLocal = SessionLocal  # módulo
    _sql_mod.SqlDocumentStorage.__init__.__defaults__ = (SessionLocal,)
    _db_base.engine = engine  # (opcional, por coherencia)

    # 2) Leemos CSV
    csv_path = Path(settings.faq_csv)
    if not csv_path.is_file():
        print(f"[ERR] CSV file not found at {csv_path}")
        sys.exit(1)

    texts = []
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter=DELIMITER)
        if settings.csv_has_header:
            next(reader, None)
        for i, row in enumerate(reader, 1):
            if len(row) < 2:
                print(f"[WARN] Row {i} skipped (len={len(row)}): {row}")
                continue
            texts.append(f"{row[0].strip()} {row[1].strip()}")
    if not texts:
        print("[ERR] No texts found in CSV.")
        sys.exit(1)

    print(f"[INFO] Parsed {len(texts)} documents from CSV.")

    # 3) Invocamos ETL con nuestro SessionLocal freshly-built
    doc_repo = SqlDocumentStorage(session_factory=SessionLocal)
    vector_repo = FaissVectorStorage(
        index_path=settings.index_path, id_map_path=settings.id_map_path
    )
    embedder = SentenceTransformerEmbedder(model_name=settings.st_embedding_model)
    etl = ETLService(doc_repo, vector_repo, embedder)
    ids = etl.ingest(texts)

    print(f"[OK] Ingested {len(ids)} docs into SQL and FAISS.")


if __name__ == "__main__":
    main()

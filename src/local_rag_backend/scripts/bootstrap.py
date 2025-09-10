# scripts/bootstrap.py

import sys
from importlib import resources
from pathlib import Path

# IMPORTS PARA BD DINÁMICA
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from local_rag_backend.core.services.etl import ETLService
from local_rag_backend.core.services.ingestion import IngestionPipeline, default_chunker
from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
    SentenceTransformerEmbedder,
)
from local_rag_backend.infrastructure.ingestion.loaders.csv_loader import CSVLoader
from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
from local_rag_backend.infrastructure.persistence.sqlalchemy import models  # noqa: F401
from local_rag_backend.infrastructure.persistence.sqlalchemy.base import Base
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage
from local_rag_backend.settings import settings

DELIMITER = ";"


def main() -> None:
    # 1) Creamos engine y sesión basados en la URL actualizada
    engine = create_engine(
        settings.sqlite_url, connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    # 2) Determine CSV path (local first, then packaged)
    csv_path = Path(settings.faq_csv)
    if not csv_path.is_file():
        pkg_csv = resources.files("local_rag_backend.data").joinpath("faq.csv")
        csv_path = Path(pkg_csv)

    doc_repo = SqlDocumentStorage(session_factory=SessionLocal)

    if settings.retrieval_mode in ["dense", "hybrid"]:
        embedder = SentenceTransformerEmbedder(model_name=settings.st_embedding_model)
        vector_repo = FaissVectorStorage(
            index_path=settings.index_path,
            id_map_path=settings.id_map_path,
            dim=embedder.dim,
        )
        etl = ETLService(doc_repo, vector_repo, embedder)
        loader = CSVLoader(csv_path, delimiter=DELIMITER, has_header=settings.csv_has_header)
        pipeline = IngestionPipeline(
            loader, 
            etl,
            chunk=default_chunker(settings.ingest_chunk_chars, settings.ingest_chunk_overlap)
        )
        ids = pipeline.run()
        print(f"[OK] Ingested {len(ids)} docs into SQL and FAISS.")
    else:
        # modo sparse: simple (sin embeddings)
        loader = CSVLoader(csv_path, delimiter=DELIMITER, has_header=settings.csv_has_header)
        texts = [li.text for li in loader.load()]
        ids = doc_repo.store_documents(texts)
        print(f"[OK] Ingested {len(ids)} docs into SQL only (sparse mode).")


if __name__ == "__main__":
    main()

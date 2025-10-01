"""
RAG Prototype - Intrinsical-AI (c) 2025
Author: Pablo Pintor
License: MIT

Bootstrap script for ingesting CSV data into the database and FAISS index.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

# IMPORTS FOR DYNAMIC DB
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from local_rag_backend.core.services.etl import ETLService
from local_rag_backend.core.services.ingestion import (
    IngestionPipeline,
    default_chunker,
    default_formatter,
    default_preprocess,
)
from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
    SentenceTransformerEmbedder,
)
from local_rag_backend.infrastructure.ingestion.loaders.csv_loader import CSVLoader
from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
from local_rag_backend.infrastructure.persistence.sqlalchemy import models  # noqa: F401
from local_rag_backend.infrastructure.persistence.sqlalchemy.base import Base
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage
from local_rag_backend.settings import settings as default_settings

DELIMITER = ";"


def main(csv_path: str | Path | None = None, **kwargs: Any) -> None:
    if "settings" not in kwargs:
        kwargs["settings"] = default_settings
    settings = kwargs["settings"]
    # 1) Create engine and session based on the updated URL
    engine = create_engine(settings.sqlite_url, connect_args={"check_same_thread": False})
    session_local = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    # 2) Determine CSV path (local first, then packaged)
    csv_path_obj = Path(settings.faq_csv) if csv_path is None else Path(csv_path)
    if not csv_path_obj.is_file():
        pkg_csv = resources.files("local_rag_backend.data").joinpath("faq.csv")
        csv_path_obj = Path(str(pkg_csv))

    doc_repo = SqlDocumentStorage(session_factory=session_local)

    if settings.retrieval_mode in ["dense", "hybrid"]:
        embedder = SentenceTransformerEmbedder(model_name=settings.st_embedding_model)
        vector_repo = FaissVectorStorage(
            index_path=settings.index_path,
            id_map_path=settings.id_map_path,
            dim=embedder.dim,
        )
        etl = ETLService(doc_repo, vector_repo, embedder)
        loader = CSVLoader(csv_path_obj, delimiter=DELIMITER, has_header=settings.csv_has_header)
        pipeline = IngestionPipeline(
            loader,
            etl,
            chunk_fn=default_chunker(settings.ingest_chunk_chars, settings.ingest_chunk_overlap),
        )
        chunk_count = pipeline.run()
        print(f"[OK] Ingested {chunk_count} docs into SQL and FAISS.")
    else:
        # sparse mode: apply the same preprocessing + chunking + formatting pipeline,
        # but storing only in SQL (no embeddings / FAISS)
        loader = CSVLoader(csv_path_obj, delimiter=DELIMITER, has_header=settings.csv_has_header)
        chunk = default_chunker(settings.ingest_chunk_chars, settings.ingest_chunk_overlap)

        buf: list[str] = []
        ids: list[int] = []
        batch = 128
        for item in loader.load():
            clean = default_preprocess(item.text, dict(item.metadata) if item.metadata else None)
            for c in chunk(clean, dict(item.metadata) if item.metadata else None):
                buf.append(default_formatter(c, dict(item.metadata) if item.metadata else None))
                if len(buf) >= batch:
                    ids += list(doc_repo.store_documents(buf))
                    buf.clear()
        if buf:
            ids += list(doc_repo.store_documents(buf))

        print(f"[OK] Ingested {len(ids)} docs into SQL only (sparse mode).")


if __name__ == "__main__":
    main()

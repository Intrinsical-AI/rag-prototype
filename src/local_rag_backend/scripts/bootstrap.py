# scripts/bootstrap.py
"""
Bootstrap script for ingesting CSV data into the database and FAISS index.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

# IMPORTS FOR DYNAMIC DB
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from local_rag_backend.app.composition import build_dense_embedder_from_settings
from local_rag_backend.core.services.etl import ETLService
from local_rag_backend.core.services.ingestion import (
    IngestionPipeline,
    build_chunk_fn_from_settings,
    build_preprocess_fn_from_settings,
    default_formatter,
)
from local_rag_backend.infrastructure.ingestion.loaders.csv_loader import CSVLoader
from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
from local_rag_backend.infrastructure.persistence.sqlalchemy import (
    base as db_base,
    models as _models,
)
from local_rag_backend.infrastructure.persistence.sqlalchemy.base import Base
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage
from local_rag_backend.settings import settings as default_settings

DELIMITER = ";"
_ = _models

if TYPE_CHECKING:
    from collections.abc import Callable

    from local_rag_backend.core.ports import EmbedderPort


def _run_with_multi_store_write_lock(settings: Any, operation: Callable[[], Any]) -> Any:
    from local_rag_backend.core.services.write_lock import multi_store_write_lock

    with multi_store_write_lock(coordination_dir=settings.get_coordination_dir()):
        return operation()


def main(csv_path: str | Path | None = None, **kwargs: Any) -> None:
    if "settings" not in kwargs:
        kwargs["settings"] = default_settings
    settings = kwargs["settings"]
    # Ensure data dir exists before touching SQLite.
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    # 1) Create engine and session based on the updated URL
    engine = create_engine(settings.sqlite_url, connect_args={"check_same_thread": False})
    session_local = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)
    # Protect multi-store integrity: migrate legacy SQLite schemas to AUTOINCREMENT if needed.
    db_base.ensure_sqlite_documents_autoincrement(
        engine_to_use=engine, id_map_path=str(settings.id_map_path)
    )
    db_base.ensure_sqlite_documents_identity_columns(engine_to_use=engine)

    # 2) Determine CSV path (explicit -> settings -> repo fallback)
    csv_path_obj = Path(settings.faq_csv) if csv_path is None else Path(csv_path)
    if not csv_path_obj.is_file():
        # Repo checkout fallback (useful in development; not guaranteed in installed distributions)
        repo_csv = Path(__file__).resolve().parents[3] / "data" / "faq.csv"
        if repo_csv.is_file():
            csv_path_obj = repo_csv
        else:
            raise FileNotFoundError(
                f"FAQ CSV not found at {csv_path_obj}. Set FAQ_CSV to a valid path."
            )

    doc_repo = SqlDocumentStorage(session_factory=session_local)

    if settings.retrieval_mode in ["dense", "hybrid"]:
        embedder: EmbedderPort
        embedder = build_dense_embedder_from_settings(settings_obj=settings)
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
            preprocess_fn=build_preprocess_fn_from_settings(settings),
            chunk_fn=build_chunk_fn_from_settings(settings),
        )
        chunk_count = int(_run_with_multi_store_write_lock(settings, pipeline.run))
        print(f"[OK] Ingested {chunk_count} docs into SQL and FAISS.")
    else:
        # sparse mode: apply the same preprocessing + chunking + formatting pipeline,
        # but storing only in SQL (no embeddings / FAISS)
        loader = CSVLoader(csv_path_obj, delimiter=DELIMITER, has_header=settings.csv_has_header)
        preprocess_fn = build_preprocess_fn_from_settings(settings)
        chunk = build_chunk_fn_from_settings(settings)

        def _ingest_sparse_locked() -> list[int]:
            buf: list[str] = []
            ids: list[int] = []
            batch = 128
            for item in loader.load():
                clean = preprocess_fn(item.text, dict(item.metadata) if item.metadata else None)
                for c in chunk(clean, dict(item.metadata) if item.metadata else None):
                    buf.append(default_formatter(c, dict(item.metadata) if item.metadata else None))
                    if len(buf) >= batch:
                        ids += list(doc_repo.store_documents(buf))
                        buf.clear()
            if buf:
                ids += list(doc_repo.store_documents(buf))
            return ids

        ids = list(_run_with_multi_store_write_lock(settings, _ingest_sparse_locked))

        print(f"[OK] Ingested {len(ids)} docs into SQL only (sparse mode).")


if __name__ == "__main__":
    main()

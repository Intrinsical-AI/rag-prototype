"""Internal sample-data ingestion implementation used by bootstrap flows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from local_rag_backend.infrastructure.persistence.sql import (
    base as db_base,
    models as _models,
)
from local_rag_backend.infrastructure.persistence.sql.alchemy_engine import SqlDocumentStorage
from local_rag_backend.infrastructure.persistence.vector.storage import VectorStorage
from local_rag_backend.settings import settings as default_settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from local_rag_backend.core.ports import EmbedderPort

DELIMITER = ";"
_ = _models

logger = logging.getLogger(__name__)


def _run_with_multi_store_write_lock(settings_obj: Any, operation: Callable[[], Any]) -> Any:
    from local_rag_backend.core.services.write_lock import multi_store_write_lock

    with multi_store_write_lock(coordination_dir=settings_obj.get_coordination_dir()):
        return operation()


def _resolve_csv_path(*, csv_path: str | Path | None, settings_obj: Any) -> Path:
    csv_path_obj = Path(settings_obj.faq_csv) if csv_path is None else Path(csv_path)
    if csv_path_obj.is_file():
        return csv_path_obj

    repo_csv = Path(__file__).resolve().parents[3] / "data" / "faq.csv"
    if repo_csv.is_file():
        return repo_csv
    raise FileNotFoundError(f"FAQ CSV not found at {csv_path_obj}. Set FAQ_CSV to a valid path.")


def run_sample_data_ingestion(
    csv_path: str | Path | None = None,
    *,
    settings_obj: Any = default_settings,
    schema_error_message: str | None = None,
) -> int:
    """
    Ingest sample CSV into SQLite and (for dense/hybrid) FAISS.

    Returns the number of ingested chunks/documents.
    """
    settings_obj.data_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(settings_obj.sqlite_url, connect_args={"check_same_thread": False})
    session_local = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    try:
        db_base.ensure_sqlite_schema_compatible(
            engine_to_use=engine, id_map_path=str(settings_obj.id_map_path)
        )
    except Exception as e:
        if schema_error_message:
            raise RuntimeError(schema_error_message) from e
        raise

    csv_path_obj = _resolve_csv_path(csv_path=csv_path, settings_obj=settings_obj)
    doc_repo = SqlDocumentStorage(session_factory=session_local)

    if settings_obj.retrieval_mode in ("dense", "hybrid"):
        embedder: EmbedderPort = build_dense_embedder_from_settings(settings_obj=settings_obj)
        vector_repo = VectorStorage(
            index_path=settings_obj.index_path,
            id_map_path=settings_obj.id_map_path,
            dim=embedder.dim,
            settings_obj=settings_obj,
        )
        etl = ETLService(doc_repo, vector_repo, embedder)
        loader = CSVLoader(
            csv_path_obj, delimiter=DELIMITER, has_header=settings_obj.csv_has_header
        )
        pipeline = IngestionPipeline(
            loader,
            etl,
            preprocess_fn=build_preprocess_fn_from_settings(settings_obj),
            chunk_fn=build_chunk_fn_from_settings(settings_obj),
        )
        chunk_count = int(_run_with_multi_store_write_lock(settings_obj, pipeline.run))
        logger.info("Ingested %d docs into SQL and FAISS.", chunk_count)
        return chunk_count

    loader = CSVLoader(csv_path_obj, delimiter=DELIMITER, has_header=settings_obj.csv_has_header)
    preprocess_fn = build_preprocess_fn_from_settings(settings_obj)
    chunk = build_chunk_fn_from_settings(settings_obj)

    def _ingest_sparse_locked() -> list[str]:
        buf: list[str] = []
        ids: list[str] = []
        batch = settings_obj.ingest_batch_size
        for item in loader.load():
            metadata = dict(item.metadata) if item.metadata else None
            clean = preprocess_fn(item.text, metadata)
            for c in chunk(clean, metadata):
                buf.append(default_formatter(c, metadata))
                if len(buf) >= batch:
                    ids += list(doc_repo.store_documents(buf))
                    buf.clear()
        if buf:
            ids += list(doc_repo.store_documents(buf))
        return ids

    ids = list(_run_with_multi_store_write_lock(settings_obj, _ingest_sparse_locked))
    logger.info("Ingested %d docs into SQL only (sparse mode).", len(ids))
    return len(ids)

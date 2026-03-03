"""Internal sample-data ingestion implementation used by bootstrap flows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from local_rag_backend.composition.adapters import build_dense_embedder_from_settings
from local_rag_backend.composition.container import AppContainer
from local_rag_backend.core.services.chunking import chunk_chars_v1
from local_rag_backend.core.services.ingestion import (
    build_preprocess_fn_from_settings,
    default_formatter,
    stable_lineage_metadata,
)
from local_rag_backend.core.use_cases.docs_mutation import (
    MutationCoordinator,
    MutationIntent,
    MutationUpsertInput,
)
from local_rag_backend.infrastructure.ingestion.loaders.csv_loader import CSVLoader
from local_rag_backend.infrastructure.persistence.sql import (
    SqlDocumentStorage,
    SystemStateStorage,
    base as db_base,
    models as _models,
)
from local_rag_backend.infrastructure.persistence.vector.storage import VectorStorage
from local_rag_backend.settings import settings as default_settings

if TYPE_CHECKING:
    from local_rag_backend.core.ports.contracts import DocsMutationPorts

DELIMITER = ";"
_ = _models

logger = logging.getLogger(__name__)


def _resolve_csv_path(*, csv_path: str | Path | None, settings_obj: Any) -> Path:
    csv_path_obj = Path(settings_obj.faq_csv) if csv_path is None else Path(csv_path)
    if csv_path_obj.is_file():
        return csv_path_obj

    repo_csv = Path(__file__).resolve().parents[3] / "data" / "faq.csv"
    if repo_csv.is_file():
        return repo_csv
    raise FileNotFoundError(f"FAQ CSV not found at {csv_path_obj}. Set FAQ_CSV to a valid path.")


def _build_bootstrap_container(
    *,
    settings_obj: Any,
    session_local: sessionmaker[Any],
) -> AppContainer:
    # Import at runtime so tests can monkeypatch lock behavior.
    from local_rag_backend.infrastructure.concurrency.locks.write_lock import (
        multi_store_write_lock,
    )

    return AppContainer.from_settings(
        settings_obj,
        doc_repo_factory=lambda: SqlDocumentStorage(session_factory=session_local),
        build_upsert_doc=SqlDocumentStorage.UpsertDoc,
        vector_repo_factory=lambda **kwargs: VectorStorage(settings_obj=settings_obj, **kwargs),
        mutation_uow_factory=lambda: db_base.session_uow(session_factory=session_local),
        system_state_factory=lambda: SystemStateStorage(session_factory=session_local),
        write_lock=multi_store_write_lock,
    )


def _coerce_row_index(value: Any, *, fallback: int) -> int:
    try:
        out = int(value)
        return out if out > 0 else fallback
    except Exception:
        return fallback


def _build_bootstrap_mutation_intent(
    *,
    csv_path_obj: Path,
    settings_obj: Any,
    ports: DocsMutationPorts,
) -> MutationIntent:
    loader = CSVLoader(csv_path_obj, delimiter=DELIMITER, has_header=settings_obj.csv_has_header)
    preprocess_fn = build_preprocess_fn_from_settings(settings_obj)

    source_id = str(csv_path_obj.resolve())
    external_id_prefix = f"bootstrap:{source_id}:"
    desired_external_ids: set[str] = set()
    items_by_external_id: dict[str, MutationUpsertInput] = {}

    for fallback_row_index, item in enumerate(loader.load(), start=1):
        metadata_base = dict(item.metadata) if item.metadata else {}
        metadata_base["_lineage"] = stable_lineage_metadata(item.lineage)
        metadata_base.setdefault("source", source_id)

        row_index = _coerce_row_index(metadata_base.get("row_index"), fallback=fallback_row_index)
        part_id = f"row-{row_index}"

        processed = preprocess_fn(item.text, metadata_base)
        chunks = chunk_chars_v1(
            processed,
            max_chars=settings_obj.ingest_chunk_chars,
            overlap=settings_obj.ingest_chunk_overlap,
        )
        for chunk in chunks:
            chunk_index = int(chunk.chunk_index)
            external_id = f"{external_id_prefix}part={part_id}:chunk={chunk_index}"
            desired_external_ids.add(external_id)

            metadata_chunk = dict(metadata_base)
            metadata_chunk["part_id"] = part_id
            metadata_chunk["chunk_index"] = chunk_index
            metadata_chunk["chunk_start_char"] = int(chunk.start_char)
            metadata_chunk["chunk_end_char"] = int(chunk.end_char)
            metadata_chunk["parent_doc_id"] = f"{external_id_prefix}part={part_id}"

            items_by_external_id[external_id] = MutationUpsertInput(
                external_id=external_id,
                content=default_formatter(chunk.text, metadata_chunk),
                source_id=source_id,
                metadata=metadata_chunk,
            )

    desired_ids_sorted = sorted(desired_external_ids)
    doc_repo = ports.doc_repo_factory()

    tombstoned: set[str] = set()
    if hasattr(doc_repo, "get_tombstoned_external_ids"):
        tombstoned = set(doc_repo.get_tombstoned_external_ids(desired_ids_sorted))

    upserts = tuple(
        items_by_external_id[eid] for eid in desired_ids_sorted if eid not in tombstoned
    )

    stale_ids: list[str] = []
    if hasattr(doc_repo, "list_ids_by_external_id_prefix"):
        existing = doc_repo.list_ids_by_external_id_prefix(external_id_prefix)
        stale_ids = sorted(
            str(doc_id) for doc_id, ext_id in existing if str(ext_id) not in desired_external_ids
        )

    return MutationIntent(
        op_id="",
        upserts=upserts,
        delete_ids=tuple(stale_ids),
        source="cli:bootstrap",
    )


def run_sample_data_ingestion(
    csv_path: str | Path | None = None,
    *,
    settings_obj: Any = default_settings,
    schema_error_message: str | None = None,
) -> int:
    """
    Ingest sample CSV into SQLite and (for dense/hybrid) vector index.

    Returns the number of processed chunks (inserted + updated + unchanged).
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
    container = _build_bootstrap_container(settings_obj=settings_obj, session_local=session_local)
    mutation_bundle = container.build_docs_mutation_bundle(
        build_embedder=lambda: build_dense_embedder_from_settings(settings_obj=settings_obj),
    )

    intent = _build_bootstrap_mutation_intent(
        csv_path_obj=csv_path_obj,
        settings_obj=settings_obj,
        ports=mutation_bundle.ports,
    )
    if not intent.upserts and not intent.delete_ids:
        return 0

    summary = MutationCoordinator(settings_obj=settings_obj, ports=mutation_bundle.ports).execute(
        intent
    )
    processed = int(summary.inserted + summary.updated + summary.unchanged)

    if settings_obj.retrieval_mode in ("dense", "hybrid"):
        logger.info("Ingested %d docs into SQL and FAISS.", processed)
    else:
        logger.info("Ingested %d docs into SQL only (sparse mode).", processed)

    return processed

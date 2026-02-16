"""CLI entry points for Intrinsical RAG Prototype."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TypeVar

import click

from local_rag_backend import __version__
from local_rag_backend.app.composition import build_dense_embedder_from_settings
from local_rag_backend.cli_commands import (
    bootstrap_cmd,
    build_index_cmd,
    delete_docs_cmd,
    delete_external_ids_cmd,
    eval_cmd,
    ingest_cmd,
    rebuild_index_cmd,
    server_cmd,
    status_cmd,
    upsert_docs_cmd,
)
from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from local_rag_backend.core.ports import EmbedderPort

T = TypeVar("T")


def _ensure_sqlite_schema_for_cli() -> None:
    """
    Ensure SQLite schema is compatible with the current ORM mappings.

    CLI commands can be run without starting the FastAPI server, so they must
    apply the same best-effort SQLite migrations that the app does at startup.
    """
    from local_rag_backend.infrastructure.persistence.sqlalchemy import base as db_base

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db_base.Base.metadata.create_all(bind=db_base.engine)
    db_base.ensure_sqlite_documents_autoincrement(
        engine_to_use=db_base.engine, id_map_path=str(settings.id_map_path)
    )
    db_base.ensure_sqlite_documents_identity_columns(engine_to_use=db_base.engine)


@click.group()
@click.version_option(version=__version__, prog_name="rag-prototype")
def cli() -> None:
    """Intrinsical RAG Prototype - Production-ready RAG system with hexagonal architecture."""
    return None


def _run_with_multi_store_write_lock(operation: Callable[[], T]) -> T:
    from local_rag_backend.core.services.write_lock import multi_store_write_lock

    with multi_store_write_lock():
        return operation()


def _reset_rag_service_best_effort() -> None:
    from local_rag_backend.app.factory import reset_rag_service

    try:
        reset_rag_service()
    except Exception:
        return None


def _build_dense_embedder() -> EmbedderPort:
    """Build the dense/hybrid embedder based on current settings."""
    from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder
    from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
        SentenceTransformerEmbedder,
    )

    return build_dense_embedder_from_settings(
        settings_obj=settings,
        openai_embedder_factory=OpenAIEmbedder,
        st_embedder_factory=lambda model_name: SentenceTransformerEmbedder(model_name=model_name),
    )


def _batched(values: list[T], batch_size: int) -> list[list[T]]:
    size = max(1, int(batch_size))
    return [values[i : i + size] for i in range(0, len(values), size)]


cli.add_command(server_cmd)
cli.add_command(build_index_cmd)
cli.add_command(rebuild_index_cmd)
cli.add_command(delete_docs_cmd)
cli.add_command(delete_external_ids_cmd)
cli.add_command(upsert_docs_cmd)
cli.add_command(bootstrap_cmd)
cli.add_command(status_cmd)
cli.add_command(eval_cmd)
cli.add_command(ingest_cmd)


def rag_server() -> None:
    """Entry point for rag-server command."""
    cli.main(args=["server"], standalone_mode=False)


def rag_build_index() -> None:
    """Entry point for rag-build-index command."""
    cli.main(args=["build-index", *sys.argv[1:]], standalone_mode=False)


def rag_bootstrap() -> None:
    """Entry point for rag-bootstrap command."""
    cli.main(args=["bootstrap", *sys.argv[1:]], standalone_mode=False)


def rag_status() -> None:
    """Entry point for rag-status command."""
    cli.main(args=["status", *sys.argv[1:]], standalone_mode=False)


def rag_eval() -> None:
    """Entry point for rag-eval command."""
    cli.main(args=["eval", *sys.argv[1:]], standalone_mode=False)


def rag_rebuild_index() -> None:
    """Entry point for rag-rebuild-index command."""
    cli.main(args=["rebuild-index", *sys.argv[1:]], standalone_mode=False)


def rag_delete_docs() -> None:
    """Entry point for rag-delete-docs command."""
    cli.main(args=["delete-docs", *sys.argv[1:]], standalone_mode=False)


def rag_upsert_docs() -> None:
    """Entry point for rag-upsert-docs command."""
    cli.main(args=["upsert-docs", *sys.argv[1:]], standalone_mode=False)


def rag_delete_external_ids() -> None:
    """Entry point for rag-delete-external-ids command."""
    cli.main(args=["delete-external-ids", *sys.argv[1:]], standalone_mode=False)


def rag_ingest() -> None:
    """Entry point for rag-ingest command."""
    cli.main(args=["ingest", *sys.argv[1:]], standalone_mode=False)


if __name__ == "__main__":
    cli()

"""CLI entry points for Intrinsical RAG Prototype."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TypeVar

import click

from local_rag_backend import __version__
from local_rag_backend.cli_commands import (
    bootstrap_cmd,
    build_index_cmd,
    delete_docs_cmd,
    delete_external_ids_cmd,
    eval_cmd,
    ingest_cmd,
    rebuild_index_cmd,
    runtime as cli_runtime,
    server_cmd,
    status_cmd,
    upsert_docs_cmd,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from local_rag_backend.core.ports import EmbedderPort

T = TypeVar("T")


def _ensure_sqlite_schema_for_cli() -> None:
    cli_runtime.ensure_sqlite_schema_for_cli()


@click.group()
@click.version_option(version=__version__, prog_name="rag-prototype")
def cli() -> None:
    """Intrinsical RAG Prototype - Production-ready RAG system with hexagonal architecture."""
    return None


def _run_with_multi_store_write_lock(operation: Callable[[], T]) -> T:
    return cli_runtime._run_with_multi_store_write_lock(operation)


def _reset_rag_service_best_effort() -> None:
    cli_runtime._reset_rag_service_best_effort()


def _run_cli_mutation(
    operation: Callable[[], T],
    *,
    use_lock: bool = True,
    ensure_schema: bool = True,
) -> T:
    return cli_runtime.run_cli_mutation(
        operation,
        use_lock=use_lock,
        ensure_schema=ensure_schema,
    )


def _build_dense_embedder() -> EmbedderPort:
    return cli_runtime.build_dense_embedder()


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

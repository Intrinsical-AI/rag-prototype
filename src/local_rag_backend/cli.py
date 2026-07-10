"""CLI entry points for Intrinsical RAG Prototype."""

from __future__ import annotations

import sys

import click

from local_rag_backend import __version__
from local_rag_backend.cli_commands import (
    bootstrap_cmd,
    eval_batch_cmd,
    eval_cmd,
    eval_compare_cmd,
    import_canonical_cmd,
    ingest_cmd,
    mutate_docs_cmd,
    rebuild_index_cmd,
    server_cmd,
    status_cmd,
)


@click.group()
@click.version_option(version=__version__, prog_name="rag-prototype")
def cli() -> None:
    """Intrinsical RAG Prototype - Production-ready RAG system with hexagonal architecture."""
    return None


cli.add_command(server_cmd)
cli.add_command(rebuild_index_cmd)
cli.add_command(mutate_docs_cmd)
cli.add_command(import_canonical_cmd)
cli.add_command(bootstrap_cmd)
cli.add_command(status_cmd)
cli.add_command(eval_cmd)
cli.add_command(eval_batch_cmd)
cli.add_command(eval_compare_cmd)
cli.add_command(ingest_cmd)


def _dispatch_entrypoint(*, command: str, include_argv: bool) -> None:
    args = [command]
    if include_argv:
        args.extend(sys.argv[1:])
    cli.main(args=args, standalone_mode=False)


def rag_server() -> None:
    """Entry point for rag-server command."""
    _dispatch_entrypoint(command="server", include_argv=False)


def rag_bootstrap() -> None:
    """Entry point for rag-bootstrap command."""
    _dispatch_entrypoint(command="bootstrap", include_argv=True)


def rag_status() -> None:
    """Entry point for rag-status command."""
    _dispatch_entrypoint(command="status", include_argv=True)


def rag_eval() -> None:
    """Entry point for rag-eval command."""
    _dispatch_entrypoint(command="eval", include_argv=True)


def rag_eval_batch() -> None:
    """Entry point for rag-eval-batch command."""
    _dispatch_entrypoint(command="eval-batch", include_argv=True)


def rag_eval_compare() -> None:
    """Entry point for rag-eval-compare command."""
    _dispatch_entrypoint(command="eval-compare", include_argv=True)


def rag_rebuild_index() -> None:
    """Entry point for rag-rebuild-index command."""
    _dispatch_entrypoint(command="rebuild-index", include_argv=True)


def rag_ingest() -> None:
    """Entry point for rag-ingest command."""
    _dispatch_entrypoint(command="ingest", include_argv=True)


def rag_mutate_docs() -> None:
    """Entry point for rag-mutate-docs command."""
    _dispatch_entrypoint(command="mutate-docs", include_argv=True)


def rag_import_canonical() -> None:
    """Entry point for rag-import-canonical command."""
    _dispatch_entrypoint(command="import-canonical", include_argv=True)


if __name__ == "__main__":
    cli()

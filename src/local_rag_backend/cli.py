"""CLI entry points for Intrinsical RAG Prototype."""

from __future__ import annotations

import sys

import click

from local_rag_backend import __version__
from local_rag_backend.cli_commands import (
    bootstrap_cmd,
    eval_cmd,
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
cli.add_command(bootstrap_cmd)
cli.add_command(status_cmd)
cli.add_command(eval_cmd)
cli.add_command(ingest_cmd)


def rag_server() -> None:
    """Entry point for rag-server command."""
    cli.main(args=["server"], standalone_mode=False)


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


def rag_ingest() -> None:
    """Entry point for rag-ingest command."""
    cli.main(args=["ingest", *sys.argv[1:]], standalone_mode=False)


def rag_mutate_docs() -> None:
    """Entry point for rag-mutate-docs command."""
    cli.main(args=["mutate-docs", *sys.argv[1:]], standalone_mode=False)


if __name__ == "__main__":
    cli()

"""Canonical RAG command tree; every installed operation belongs to this entry point."""

from __future__ import annotations

import json

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
from local_rag_backend.cli_commands.snapshot import snapshot_cmd
from local_rag_backend.composition.factory import reset_app_context
from local_rag_backend.composition.operations import search
from local_rag_backend.settings import get_settings


@click.group()
@click.version_option(version=__version__, prog_name="rag")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Manage documents, retrieval, generation, evaluation and transports."""
    ctx.call_on_close(reset_app_context)


@cli.group()
def docs() -> None:
    """Create, ingest, import and mutate documents."""


docs.add_command(bootstrap_cmd, name="bootstrap")
docs.add_command(ingest_cmd, name="ingest")
docs.add_command(import_canonical_cmd, name="import")
docs.add_command(mutate_docs_cmd, name="mutate")


@cli.group()
def index() -> None:
    """Maintain the vector index."""


index.add_command(rebuild_index_cmd, name="rebuild")


@cli.group()
def query() -> None:
    """Search or generate a response from the private corpus."""


@query.command("search")
@click.argument("question")
@click.option("-k", type=click.IntRange(1, 100), default=3, show_default=True)
@click.option("--filters-json", default="[]", help="JSON list of field/values filters.")
def search_cmd(question: str, k: int, filters_json: str) -> None:
    """Retrieve sources without constructing a generator or saving history."""
    from local_rag_backend.core.services.retrieval_filters import parse_filters

    try:
        filters = parse_filters(json.loads(filters_json))
        click.echo(
            json.dumps(search(settings=get_settings(), question=question, k=k, filters=filters))
        )
    except (ValueError, OSError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc


@query.command("ask")
@click.argument("question")
@click.option("-k", type=click.IntRange(1, 10), default=3, show_default=True)
def ask_cmd(question: str, k: int) -> None:
    """Generate an answer and save its normal configured history."""
    from local_rag_backend.composition.commands import get_container
    from local_rag_backend.composition.operations import document_to_payload

    container = get_container()
    result = container.build_rag_service().ask(
        question, top_k=k, retrieval_mode=container.settings_obj.retrieval_mode
    )
    click.echo(
        json.dumps(
            {
                "answer": result["answer"],
                "sources": [
                    {"document": document_to_payload(document), "score": score}
                    for document, score in zip(result["docs"], result["scores"], strict=True)
                ],
            }
        )
    )


@cli.group()
def serve() -> None:
    """Run an HTTP or native stdio MCP transport."""


serve.add_command(server_cmd, name="http")


@serve.command("mcp")
def mcp_cmd() -> None:
    """Run the optional MCP transport."""
    from local_rag_backend.mcp_server import main

    main()


@cli.group(name="eval")
def evaluation() -> None:
    """Run, batch and compare retrieval evaluations."""


evaluation.add_command(eval_cmd, name="run")
evaluation.add_command(eval_batch_cmd, name="batch")
evaluation.add_command(eval_compare_cmd, name="compare")
cli.add_command(status_cmd, name="status")
cli.add_command(snapshot_cmd, name="snapshot")


if __name__ == "__main__":
    cli()

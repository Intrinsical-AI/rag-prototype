"""Explicit snapshot preparation is a writer command, outside MCP query flows."""

from __future__ import annotations

import json
from pathlib import Path

import click
from sqlalchemy.exc import SQLAlchemyError

from local_rag_backend.infrastructure.persistence.read_snapshot import prepare_snapshot
from local_rag_backend.settings import get_settings


@click.command("snapshot")
@click.argument("destination", type=click.Path(path_type=Path))
def snapshot_cmd(destination: Path) -> None:
    """Prepare a fresh sealed documents-only snapshot at DESTINATION."""
    try:
        manifest = prepare_snapshot(get_settings(), destination)
    except (OSError, ValueError, SQLAlchemyError) as exc:
        raise click.ClickException(
            "Snapshot preparation refused; check source consistency and use a fresh destination"
        ) from exc
    click.echo(json.dumps(manifest.model_dump(mode="json"), sort_keys=True))

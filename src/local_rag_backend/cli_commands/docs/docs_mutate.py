from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import click

from local_rag_backend.composition.commands import get_container
from local_rag_backend.composition.operations import mutate_documents
from local_rag_backend.core.errors import MutationRecoveryRequiredError


def _read_payload(payload_json: Path) -> dict[str, Any]:
    payload = json.loads(payload_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--json must contain a JSON object payload")
    return cast("dict[str, Any]", payload)


@click.command("mutate")
@click.option(
    "--json",
    "payload_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to mutation payload JSON ({op_id?, upserts?, delete_ids?, delete_external_ids?}).",
)
def mutate_docs_cmd(payload_json: Path) -> None:
    """Run one unified docs mutation request through the durable coordinator."""
    try:
        payload = _read_payload(payload_json)
        summary = mutate_documents(
            container=get_container(), payload=payload, source="cli:docs:mutate"
        )
        click.echo(
            "[OK] Mutation committed. "
            f"op_id={summary.op_id} inserted={summary.inserted} updated={summary.updated} "
            f"unchanged={summary.unchanged} deleted_sql={summary.deleted_sql} "
            f"deleted_index={summary.deleted_index} tombstoned={summary.tombstoned}"
        )
    except MutationRecoveryRequiredError as exc:
        click.echo(f"[ERROR] Mutation requires recovery: {exc}", err=True)
        raise SystemExit(1) from None
    except Exception as e:
        click.echo(f"[ERROR] Error running docs mutation: {e}", err=True)
        raise SystemExit(1)

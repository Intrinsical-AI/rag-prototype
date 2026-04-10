from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import click
from pydantic import ValidationError

from local_rag_backend.cli_commands.runtime import (
    build_dense_embedder,
    get_cli_container,
    run_cli_mutation,
)
from local_rag_backend.core.services.docs_mutation_transport import (
    build_docs_mutation_intent_from_raw,
)
from local_rag_backend.core.use_cases.docs_mutation import MutationCoordinator, MutationIntent
from local_rag_backend.core.use_cases.results import MutationSummary
from local_rag_backend.settings import settings


def _read_payload(payload_json: Path) -> dict[str, Any]:
    payload = json.loads(payload_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--json must contain a JSON object payload")
    return cast("dict[str, Any]", payload)

def _build_intent(payload: dict[str, Any]) -> MutationIntent:
    try:
        return build_docs_mutation_intent_from_raw(payload, source="cli:docs:mutate")
    except (ValidationError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


@click.command("mutate-docs")
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
        intent = _build_intent(payload)
        container = get_cli_container()
        mutation_bundle = container.build_docs_mutation_bundle(
            build_embedder=build_dense_embedder,
        )
        coordinator = MutationCoordinator(settings_obj=settings, ports=mutation_bundle.ports)

        def _run_sync() -> MutationSummary:
            return coordinator.execute(intent)

        summary = run_cli_mutation(_run_sync, use_lock=False)
        click.echo(
            "[OK] Mutation committed. "
            f"op_id={summary.op_id} inserted={summary.inserted} updated={summary.updated} "
            f"unchanged={summary.unchanged} deleted_sql={summary.deleted_sql} "
            f"deleted_index={summary.deleted_index} tombstoned={summary.tombstoned}"
        )
    except Exception as e:
        click.echo(f"[ERROR] Error running docs mutation: {e}", err=True)
        raise SystemExit(1)

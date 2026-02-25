from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import click

from local_rag_backend.app.application.docs_mutation import (
    MutationCoordinator,
    MutationIntent,
    MutationUpsertInput,
)
from local_rag_backend.app.contracts.results import MutationSummary
from local_rag_backend.app.wiring.mutation_ports import build_docs_mutation_ports
from local_rag_backend.cli_commands.runtime import build_dense_embedder, run_cli_mutation
from local_rag_backend.settings import settings


def _read_payload(payload_json: Path) -> dict[str, Any]:
    payload = json.loads(payload_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--json must contain a JSON object payload")
    return cast("dict[str, Any]", payload)


def _build_intent(payload: dict[str, Any]) -> MutationIntent:
    upserts_raw = payload.get("upserts") or []
    delete_ids_raw = payload.get("delete_ids") or []
    delete_external_ids_raw = payload.get("delete_external_ids") or []

    if not isinstance(upserts_raw, list):
        raise ValueError("payload.upserts must be a list")
    if not isinstance(delete_ids_raw, list):
        raise ValueError("payload.delete_ids must be a list")
    if not isinstance(delete_external_ids_raw, list):
        raise ValueError("payload.delete_external_ids must be a list")

    upserts: list[MutationUpsertInput] = []
    for item in upserts_raw:
        if not isinstance(item, dict):
            raise ValueError("each payload.upserts item must be an object")
        md = item.get("metadata")
        upserts.append(
            MutationUpsertInput(
                external_id=str(item.get("external_id") or "").strip(),
                content=str(item.get("content") or "").strip(),
                source_id=(
                    str(item.get("source_id")) if item.get("source_id") is not None else None
                ),
                metadata=(cast("dict[str, Any]", md) if isinstance(md, dict) else None),
            )
        )

    return MutationIntent(
        op_id=str(payload.get("op_id") or "").strip(),
        upserts=tuple(upserts),
        delete_ids=tuple(str(v) for v in delete_ids_raw),
        delete_external_ids=tuple(str(v) for v in delete_external_ids_raw),
        source="cli:docs:mutate",
    )


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
        ports = build_docs_mutation_ports(build_embedder=build_dense_embedder)
        coordinator = MutationCoordinator(settings_obj=settings, ports=ports)

        def _run_sync() -> MutationSummary:
            return coordinator.execute(intent)

        summary = run_cli_mutation(_run_sync)
        click.echo(
            "[OK] Mutation committed. "
            f"op_id={summary.op_id} inserted={summary.inserted} updated={summary.updated} "
            f"unchanged={summary.unchanged} deleted_sql={summary.deleted_sql} "
            f"deleted_index={summary.deleted_index} tombstoned={summary.tombstoned}"
        )
    except Exception as e:
        click.echo(f"[ERROR] Error running docs mutation: {e}", err=True)
        raise SystemExit(1)

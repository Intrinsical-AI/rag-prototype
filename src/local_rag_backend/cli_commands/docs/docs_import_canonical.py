from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import click

from local_rag_backend.cli_commands.runtime import get_cli_container, run_cli_mutation
from local_rag_backend.core.use_cases.docs_import_canonical import (
    CanonicalImportDocumentInput,
    CanonicalImportRequestInput,
    execute_import_canonical_sync,
)
from local_rag_backend.core.use_cases.results import CanonicalImportSummary
from local_rag_backend.settings import settings


def _read_payload(payload_json: Path) -> dict[str, Any]:
    payload = json.loads(payload_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--json must contain a JSON object payload")
    return cast("dict[str, Any]", payload)


def _build_request(
    payload: dict[str, Any],
    *,
    replace_scope: bool,
) -> CanonicalImportRequestInput:
    documents_raw = payload.get("documents") or []
    if not isinstance(documents_raw, list):
        raise ValueError("payload.documents must be a list")

    documents: list[CanonicalImportDocumentInput] = []
    for item in documents_raw:
        if not isinstance(item, dict):
            raise ValueError("each payload.documents item must be an object")
        metadata = item.get("metadata")
        documents.append(
            CanonicalImportDocumentInput(
                external_id=str(item.get("external_id") or "").strip(),
                content=str(item.get("content") or "").strip(),
                source_id=(
                    str(item.get("source_id")) if item.get("source_id") is not None else None
                ),
                metadata=(cast("dict[str, Any]", metadata) if isinstance(metadata, dict) else None),
            )
        )

    return CanonicalImportRequestInput(
        scope=str(payload.get("scope") or "").strip(),
        snapshot_id=str(payload.get("snapshot_id") or "").strip(),
        replace_scope=replace_scope,
        documents=tuple(documents),
        source="cli:docs:import-canonical",
    )


def _resolve_replace_scope(
    payload: dict[str, Any],
    *,
    replace_scope_override: bool | None,
) -> bool:
    if replace_scope_override is not None:
        return bool(replace_scope_override)
    payload_value = payload.get("replace_scope")
    if payload_value is None:
        return True
    if isinstance(payload_value, bool):
        return payload_value
    raise ValueError("payload.replace_scope must be a boolean when provided")


@click.command("import-canonical")
@click.option(
    "--json",
    "payload_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to canonical import JSON ({scope, snapshot_id, documents[]}).",
)
@click.option(
    "--replace-scope",
    "replace_scope_override",
    flag_value=True,
    default=None,
    help="Delete stale docs in the same scope after importing the snapshot.",
)
@click.option(
    "--upsert-only",
    "replace_scope_override",
    flag_value=False,
    help="Do not delete stale docs in the same scope after importing the snapshot.",
)
def import_canonical_cmd(payload_json: Path, replace_scope_override: bool | None) -> None:
    """Import/sync external canonical documents through the native mutation stack."""
    try:
        payload = _read_payload(payload_json)
        request = _build_request(
            payload,
            replace_scope=_resolve_replace_scope(
                payload,
                replace_scope_override=replace_scope_override,
            ),
        )
        container = get_cli_container()
        mutation_bundle = container.build_docs_mutation_bundle()

        def _run_sync() -> CanonicalImportSummary:
            return execute_import_canonical_sync(
                request=request,
                settings_obj=settings,
                ports=mutation_bundle.ports,
            )

        summary = run_cli_mutation(_run_sync, use_lock=True)
        click.echo(
            "[OK] Canonical import committed. "
            f"scope={summary.scope} snapshot_id={summary.snapshot_id} "
            f"inserted={summary.inserted} updated={summary.updated} "
            f"unchanged={summary.unchanged} deleted_sql={summary.deleted_sql} "
            f"deleted_index={summary.deleted_index}"
        )
    except Exception as e:
        click.echo(f"[ERROR] Error importing canonical documents: {e}", err=True)
        raise SystemExit(1)

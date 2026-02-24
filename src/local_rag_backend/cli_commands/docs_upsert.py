from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import click

from local_rag_backend.app.services import docs as docs_service
from local_rag_backend.app.services.mutation_ports import build_docs_mutation_ports
from local_rag_backend.cli_commands.runtime import build_dense_embedder, run_cli_mutation
from local_rag_backend.settings import settings


def _load_docs_payload(
    *,
    json_path: Path | None,
    external_id: str | None,
    content: str | None,
    source_id: str | None,
    metadata_json: str | None,
) -> list[dict[str, object]]:
    if json_path is not None:
        docs_payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(docs_payload, list):
            raise ValueError("--json must contain a JSON list of documents")
        return cast("list[dict[str, object]]", docs_payload)

    if not external_id or not content:
        raise ValueError(
            "Provide --json or both --external-id and --content for a single document."
        )

    md_single = None
    if metadata_json:
        md_single = json.loads(metadata_json)
        if not isinstance(md_single, dict):
            raise ValueError("--metadata-json must be a JSON object")
    return [
        {
            "external_id": external_id,
            "content": content,
            "source_id": source_id,
            "metadata": md_single,
        }
    ]


def _build_upsert_items(
    docs_payload: list[dict[str, object]],
) -> list[Any]:
    from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

    items: list[Any] = []
    for d in docs_payload:
        if not isinstance(d, dict):
            raise ValueError("Each document must be a JSON object")
        md_obj = d.get("metadata")
        md: dict[str, Any] | None = (
            cast("dict[str, Any]", md_obj) if isinstance(md_obj, dict) else None
        )
        items.append(
            SqlDocumentStorage.UpsertDoc(
                external_id=str(d.get("external_id") or "").strip(),
                content=str(d.get("content") or "").strip(),
                source_id=(str(d.get("source_id")) if d.get("source_id") is not None else None),
                metadata=md,
            )
        )
    ext_ids = [it.external_id for it in items]
    if len(set(ext_ids)) != len(ext_ids):
        raise ValueError("external_id values must be unique within the request")
    return items


@click.command("upsert-docs")
@click.option(
    "--json",
    "json_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
    help="Path to a JSON file containing a list of {external_id, content, source_id?, metadata?}.",
)
@click.option("--external-id", type=str, required=False, help="External ID for a single document.")
@click.option("--content", type=str, required=False, help="Content for a single document.")
@click.option("--source-id", type=str, required=False, help="Optional source identifier.")
@click.option(
    "--metadata-json",
    type=str,
    required=False,
    help="Optional metadata as JSON string for a single document.",
)
def upsert_docs_cmd(
    json_path: Path | None,
    external_id: str | None,
    content: str | None,
    source_id: str | None,
    metadata_json: str | None,
) -> None:
    """Upsert documents by external_id (idempotent)."""
    try:
        docs_payload = _load_docs_payload(
            json_path=json_path,
            external_id=external_id,
            content=content,
            source_id=source_id,
            metadata_json=metadata_json,
        )
        items = _build_upsert_items(docs_payload)

        ports = build_docs_mutation_ports(build_embedder=build_dense_embedder)

        def _upsert_sync() -> docs_service.UpsertDocsSummary:
            return docs_service.upsert_docs_sync(
                docs=items,
                settings_obj=settings,
                ports=ports,
            )

        summary = run_cli_mutation(_upsert_sync)
        inserted = summary.inserted
        updated = summary.updated
        unchanged = summary.unchanged
        rebuilt = summary.rebuilt_index
        click.echo(
            f"[OK] Upserted docs. inserted={inserted} updated={updated} unchanged={unchanged} rebuilt_index={rebuilt}"
        )
    except docs_service.TombstonedExternalIdsError as e:
        click.echo(f"[ERROR] Error upserting docs: {e}", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"[ERROR] Error upserting docs: {e}", err=True)
        raise SystemExit(1)

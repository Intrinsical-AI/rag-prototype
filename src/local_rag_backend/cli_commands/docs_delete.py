from __future__ import annotations

import click

from local_rag_backend.app.services import docs as docs_service
from local_rag_backend.app.services.mutation_ports import build_docs_mutation_ports
from local_rag_backend.cli_commands.runtime import build_dense_embedder, run_cli_mutation
from local_rag_backend.settings import settings


@click.command("delete-docs")
@click.argument("ids", nargs=-1, type=int)
def delete_docs_cmd(ids: tuple[int, ...]) -> None:
    """Delete documents by ID from SQLite (and FAISS in dense/hybrid mode)."""
    if not ids:
        click.echo("[ERROR] Provide one or more document IDs.", err=True)
        raise SystemExit(2)

    try:
        ports = build_docs_mutation_ports(build_embedder=build_dense_embedder)

        def _delete_sync() -> docs_service.DeleteDocsSummary:
            return docs_service.delete_docs_sync(
                ids=list(ids),
                settings_obj=settings,
                ports=ports,
            )

        summary = run_cli_mutation(_delete_sync)
        deleted_sql = summary.deleted_sql
        deleted_index = summary.deleted_index
        rebuilt = summary.rebuilt_index
        if deleted_index is None and settings.retrieval_mode not in ("dense", "hybrid"):
            click.echo(f"[OK] Deleted {deleted_sql} docs from SQL.")
            return
        click.echo(
            f"[OK] Deleted {deleted_sql} docs from SQL. "
            f"Index delete={'ok' if deleted_index is not None else 'n/a'} "
            f"rebuilt={rebuilt}."
        )
    except Exception as e:
        click.echo(f"[ERROR] Error deleting docs: {e}", err=True)
        raise SystemExit(1)


@click.command("delete-external-ids")
@click.argument("external_ids", nargs=-1, type=str)
def delete_external_ids_cmd(external_ids: tuple[str, ...]) -> None:
    """Delete documents by external_id from SQLite (and FAISS in dense/hybrid mode), adding tombstones."""
    if not external_ids:
        click.echo("[ERROR] Provide one or more external_ids.", err=True)
        raise SystemExit(2)

    try:
        ports = build_docs_mutation_ports(build_embedder=build_dense_embedder)

        def _delete_sync() -> docs_service.DeleteDocsByExternalIdSummary:
            return docs_service.delete_docs_by_external_id_sync(
                external_ids=list(external_ids),
                settings_obj=settings,
                ports=ports,
            )

        summary = run_cli_mutation(_delete_sync)
        deleted_sql = summary.deleted_sql
        deleted_index = summary.deleted_index
        missing = summary.missing_external_ids
        tombstoned = summary.tombstoned
        rebuilt = summary.rebuilt_index
        click.echo(
            f"[OK] Deleted {deleted_sql} docs by external_id. "
            f"tombstoned={tombstoned} missing={len(missing)} "
            f"index_delete={'ok' if deleted_index is not None else 'n/a'} rebuilt={rebuilt}."
        )
        if missing:
            click.echo(f"[INFO] Missing external_ids (tombstoned anyway): {missing[:10]}")
    except Exception as e:
        click.echo(f"[ERROR] Error deleting by external_id: {e}", err=True)
        raise SystemExit(1)

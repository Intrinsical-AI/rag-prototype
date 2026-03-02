from __future__ import annotations

from pathlib import Path

import click

from local_rag_backend.cli_commands.docs._ingestion_planner import (
    build_ingest_plans,
    discover_input_files,
    execute_ingest_batches,
    resolve_loader_options,
)
from local_rag_backend.cli_commands.runtime import (
    build_dense_embedder,
    get_cli_container,
    run_cli_mutation,
)
from local_rag_backend.core.use_cases.docs_mutation import MutationCoordinator
from local_rag_backend.settings import settings


@click.command("ingest")
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--recursive/--no-recursive", default=True, show_default=True)
@click.option("--follow-symlinks/--no-follow-symlinks", default=False, show_default=True)
@click.option("--max-files", type=int, default=2000, show_default=True)
@click.option("--max-file-bytes", type=int, default=2_000_000, show_default=True)
@click.option("--max-total-bytes", type=int, default=50_000_000, show_default=True)
@click.option(
    "--csv-delimiter",
    type=str,
    default="auto",
    show_default=True,
    help="CSV delimiter. Use 'auto' to sniff, or provide one of ',', ';', '\\t', '|'.",
)
@click.option("--csv-has-header/--csv-no-header", default=None)
@click.option("--sniff-bytes", type=int, default=4096, show_default=True)
@click.option("--use-magic/--no-magic", default=True, show_default=True)
@click.option(
    "--dry-run", is_flag=True, help="Discover and parse files, but don't write to DB/index."
)
def ingest_cmd(
    paths: tuple[Path, ...],
    recursive: bool,
    follow_symlinks: bool,
    max_files: int,
    max_file_bytes: int,
    max_total_bytes: int,
    csv_delimiter: str,
    csv_has_header: bool | None,
    sniff_bytes: int,
    use_magic: bool,
    dry_run: bool,
) -> None:
    """
    Ingest documents from file(s) or directory(ies) into SQLite (+ vector index for dense/hybrid).

    Supported formats (best-effort): .txt, .md, .csv.
    """
    if not paths:
        click.echo("[ERROR] Provide one or more paths (file or directory).", err=True)
        raise SystemExit(2)

    try:
        from local_rag_backend.core.services.ingestion import build_preprocess_fn_from_settings

        preprocess_fn = build_preprocess_fn_from_settings(settings)
        delimiter_opt, has_header = resolve_loader_options(
            settings_obj=settings,
            csv_delimiter=csv_delimiter,
            csv_has_header=csv_has_header,
        )

        files = discover_input_files(
            paths=paths,
            recursive=recursive,
            follow_symlinks=follow_symlinks,
            max_files=max_files,
            max_file_bytes=max_file_bytes,
            max_total_bytes=max_total_bytes,
        )
        if not files:
            click.echo("[WARN] No files found under limits. Nothing to ingest.")
            return

        container = get_cli_container()
        mutation_bundle = container.build_docs_mutation_bundle(
            build_embedder=build_dense_embedder,
        )
        ports = mutation_bundle.ports

        ingest_plans, total_files, dry_run_chunks, total_skipped = build_ingest_plans(
            files=files,
            preprocess_fn=preprocess_fn,
            build_upsert_doc=ports.build_upsert_doc,
            settings_obj=settings,
            sniff_bytes=sniff_bytes,
            use_magic=use_magic,
            delimiter_opt=delimiter_opt,
            has_header=has_header,
            dry_run=dry_run,
        )

        if dry_run:
            click.echo(
                f"[DRY-RUN] files={total_files} chunks={dry_run_chunks} skipped={total_skipped} "
                f"(limits: max_files={max_files} max_file_bytes={max_file_bytes} max_total_bytes={max_total_bytes})"
            )
            return

        doc_repo = ports.doc_repo_factory()
        coordinator = MutationCoordinator(settings_obj=settings, ports=ports)

        def _ingest_sync() -> tuple[int, int, int, int, bool]:
            return execute_ingest_batches(
                ingest_plans=ingest_plans,
                doc_repo=doc_repo,
                coordinator=coordinator,
                settings_obj=settings,
            )

        total_inserted, total_updated, total_unchanged, total_chunks, rebuilt_any = (
            run_cli_mutation(_ingest_sync, use_lock=False)
        )
        click.echo(
            f"[OK] Ingest completed. files={total_files} chunks={total_chunks} "
            f"inserted={total_inserted} updated={total_updated} unchanged={total_unchanged} "
            f"skipped={total_skipped} rebuilt_index={rebuilt_any}"
        )
    except Exception as e:
        click.echo(f"[ERROR] Error ingesting files: {e}", err=True)
        raise SystemExit(1)

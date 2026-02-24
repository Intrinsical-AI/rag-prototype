from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from local_rag_backend.cli_commands.runtime import build_dense_embedder, run_cli_mutation
from local_rag_backend.core.services.dense_upsert import (
    precompute_vectors_for_changed_items,
    sync_dense_after_upsert,
)
from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from local_rag_backend.core.ports import EmbedderPort

IngestPlan = tuple[Path, str, tuple[Any, ...], tuple[str, ...]]
BatchSyncResult = tuple[int, int, int, bool, int, int, list[tuple[Path, int]]]


def _resolve_loader_options(
    *, csv_delimiter: str, csv_has_header: bool | None
) -> tuple[str | None, bool]:
    delimiter_opt = None if csv_delimiter.strip().lower() == "auto" else csv_delimiter
    has_header = settings.csv_has_header if csv_has_header is None else bool(csv_has_header)
    return delimiter_opt, has_header


def _discover_input_files(
    *,
    paths: tuple[Path, ...],
    recursive: bool,
    follow_symlinks: bool,
    max_files: int,
    max_file_bytes: int,
    max_total_bytes: int,
) -> list[Path]:
    from local_rag_backend.infrastructure.ingestion.loaders.discovery import discover_files

    return list(
        discover_files(
            list(paths),
            recursive=recursive,
            follow_symlinks=follow_symlinks,
            max_files=max_files,
            max_file_bytes=max_file_bytes,
            max_total_bytes=max_total_bytes,
        )
    )


def _build_file_ingest_plan(
    *,
    file_path: Path,
    preprocess_fn: Callable[..., str],
    sniff_bytes: int,
    use_magic: bool,
    delimiter_opt: str | None,
    has_header: bool,
) -> IngestPlan | None:
    from local_rag_backend.core.services.chunking import chunk_chars_v1
    from local_rag_backend.core.services.ingestion import default_formatter
    from local_rag_backend.infrastructure.ingestion.loaders.factory import (
        detect_file_format,
        get_loader_for_file,
    )
    from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

    det = detect_file_format(file_path, sniff_bytes=sniff_bytes, use_magic=use_magic)
    loader = get_loader_for_file(
        file_path,
        sniff_bytes=sniff_bytes,
        use_magic=use_magic,
        csv_delimiter=delimiter_opt,
        csv_has_header=has_header,
        detection=det,
    )
    if loader is None:
        return None

    source_id = str(file_path.resolve())
    file_prefix = f"file:{source_id}:"
    items: list[Any] = []
    desired_external_ids: set[str] = set()

    for loaded in loader.load():
        md = dict(loaded.metadata) if loaded.metadata else {}
        md.setdefault("source", str(file_path))
        md.setdefault("format", det.fmt)
        md.setdefault("filename", file_path.name)

        part_id = "file"
        if "row_index" in md and md["row_index"] is not None:
            part_id = f"row-{md['row_index']}"

        parent_doc_id = f"{file_prefix}part={part_id}"
        processed = preprocess_fn(loaded.text, md)
        chunks = chunk_chars_v1(
            processed,
            max_chars=settings.ingest_chunk_chars,
            overlap=settings.ingest_chunk_overlap,
        )
        for c in chunks:
            chunk_index = int(c.chunk_index)
            chunk = c.text
            external_id = f"{file_prefix}part={part_id}:chunk={chunk_index}"
            desired_external_ids.add(external_id)

            md_chunk = dict(md)
            md_chunk["part_id"] = part_id
            md_chunk["chunk_index"] = chunk_index
            md_chunk["chunk_start_char"] = int(c.start_char)
            md_chunk["chunk_end_char"] = int(c.end_char)
            md_chunk["parent_doc_id"] = parent_doc_id
            content = default_formatter(chunk, md_chunk)
            items.append(
                SqlDocumentStorage.UpsertDoc(
                    external_id=external_id,
                    content=content,
                    source_id=source_id,
                    metadata=md_chunk,
                )
            )

    if not items:
        return None
    return (file_path, file_prefix, tuple(items), tuple(desired_external_ids))


def _build_ingest_plans(
    *,
    files: list[Path],
    preprocess_fn: Callable[..., str],
    sniff_bytes: int,
    use_magic: bool,
    delimiter_opt: str | None,
    has_header: bool,
    dry_run: bool,
) -> tuple[list[IngestPlan], int, int, int]:
    plans: list[IngestPlan] = []
    total_files = 0
    total_chunks = 0
    total_skipped = 0

    for file_path in files:
        plan = _build_file_ingest_plan(
            file_path=file_path,
            preprocess_fn=preprocess_fn,
            sniff_bytes=sniff_bytes,
            use_magic=use_magic,
            delimiter_opt=delimiter_opt,
            has_header=has_header,
        )
        if plan is None:
            total_skipped += 1
            continue

        total_files += 1
        if dry_run:
            total_chunks += len(plan[2])
            continue
        plans.append(plan)
    return plans, total_files, total_chunks, total_skipped


def _collect_batch_items_and_stale(
    *,
    plans_bound: tuple[IngestPlan, ...],
    doc_repo: Any,
) -> tuple[list[Any], list[int], list[tuple[Path, int]], int]:
    all_items: list[Any] = []
    stale_ids_all: list[int] = []
    stale_by_file: list[tuple[Path, int]] = []
    ingested_chunks = 0

    for file_path_bound, file_prefix_bound, items_bound, desired_bound in plans_bound:
        desired_external_ids_local = set(desired_bound)
        items_local = list(items_bound)

        tombstoned = doc_repo.get_tombstoned_external_ids(list(desired_external_ids_local))
        if tombstoned:
            desired_external_ids_local -= tombstoned
            items_local = [it for it in items_local if it.external_id not in tombstoned]

        existing = doc_repo.list_ids_by_external_id_prefix(file_prefix_bound)
        stale_ids = [doc_id for doc_id, ext in existing if ext not in desired_external_ids_local]
        if stale_ids:
            stale_ids_all.extend(stale_ids)
            stale_by_file.append((file_path_bound, len(stale_ids)))

        ingested_chunks += len(items_local)
        all_items.extend(items_local)

    return all_items, stale_ids_all, stale_by_file, ingested_chunks


def _deduplicate_items_by_external_id(items: list[Any]) -> list[Any]:
    unique_items: list[Any] = []
    seen_external_ids: set[str] = set()
    for item in items:
        if item.external_id in seen_external_ids:
            continue
        seen_external_ids.add(item.external_id)
        unique_items.append(item)
    return unique_items


def _sync_and_cleanup_dense_batch(
    *,
    results: list[Any],
    updated_content_ids: list[int],
    vectors_by_external_id: dict[str, list[float]],
    doc_repo: Any,
    embedder: EmbedderPort,
    vec: Any,
    stale_ids_unique: list[int],
) -> tuple[bool, int]:
    from local_rag_backend.core.services.maintenance import delete_documents_multi_store

    rebuilt = sync_dense_after_upsert(
        results=results,
        updated_content_ids=updated_content_ids,
        vectors_by_external_id=vectors_by_external_id,
        vec_repo=vec,
        doc_repo=doc_repo,
        embedder=embedder,
    )
    deleted_stale = 0
    if stale_ids_unique:
        deleted_sql, _, rebuilt_del = delete_documents_multi_store(
            doc_repo=doc_repo,
            ids=stale_ids_unique,
            vec_repo=vec,
            embedder=embedder,
            rebuild_on_index_failure=True,
        )
        deleted_stale = int(deleted_sql)
        rebuilt = rebuilt or rebuilt_del
    return rebuilt, deleted_stale


def _ingest_batch_sync(
    *,
    plans_bound: tuple[IngestPlan, ...],
    doc_repo: Any,
    embedder: EmbedderPort | None,
    vec: Any,
) -> BatchSyncResult:
    from local_rag_backend.core.services.maintenance import delete_documents_multi_store

    all_items, stale_ids_all, stale_by_file, ingested_chunks = _collect_batch_items_and_stale(
        plans_bound=plans_bound,
        doc_repo=doc_repo,
    )
    unique_items = _deduplicate_items_by_external_id(all_items)

    vectors_by_external_id: dict[str, list[float]] = {}
    if settings.retrieval_mode in ("dense", "hybrid") and unique_items:
        if embedder is None:
            raise RuntimeError(
                "Dense embedder is required for dense/hybrid retrieval mode but was not initialized"
            )
        vectors_by_external_id = precompute_vectors_for_changed_items(
            items=unique_items,
            doc_repo=doc_repo,
            embedder=embedder,
        )

    results: list[Any] = []
    updated_content_ids: list[int] = []
    if unique_items:
        results, _changed_content, updated_content_ids = doc_repo.upsert_documents_by_external_id(
            unique_items
        )

    inserted = sum(1 for r in results if r.action == "inserted")
    updated = sum(1 for r in results if r.action == "updated")
    unchanged = sum(1 for r in results if r.action == "unchanged")
    rebuilt = False
    deleted_stale = 0
    stale_ids_unique = sorted({int(x) for x in stale_ids_all})

    if settings.retrieval_mode in ("dense", "hybrid"):
        if embedder is None or vec is None:
            raise RuntimeError(
                "Dense embedder and vector repo are required for dense/hybrid retrieval mode"
            )
        rebuilt, deleted_stale = _sync_and_cleanup_dense_batch(
            results=results,
            updated_content_ids=updated_content_ids,
            vectors_by_external_id=vectors_by_external_id,
            doc_repo=doc_repo,
            embedder=embedder,
            vec=vec,
            stale_ids_unique=stale_ids_unique,
        )
    elif stale_ids_unique:
        deleted_sql, _, _ = delete_documents_multi_store(doc_repo=doc_repo, ids=stale_ids_unique)
        deleted_stale = int(deleted_sql)

    return (
        inserted,
        updated,
        unchanged,
        rebuilt,
        deleted_stale,
        ingested_chunks,
        stale_by_file,
    )


def _execute_ingest_batches(
    *,
    ingest_plans: list[IngestPlan],
    doc_repo: Any,
    embedder: EmbedderPort | None,
    vec: Any,
) -> tuple[int, int, int, int, bool]:
    total_inserted = 0
    total_updated = 0
    total_unchanged = 0
    total_chunks = 0
    rebuilt_any = False

    batch_size = settings.ingest_batch_size
    for i in range(0, len(ingest_plans), batch_size):
        plan_batch = ingest_plans[i : i + batch_size]
        (
            inserted,
            updated,
            unchanged,
            rebuilt,
            _deleted_stale,
            ingested_chunks,
            stale_by_file,
        ) = _ingest_batch_sync(
            plans_bound=tuple(plan_batch),
            doc_repo=doc_repo,
            embedder=embedder,
            vec=vec,
        )
        total_chunks += ingested_chunks
        total_inserted += inserted
        total_updated += updated
        total_unchanged += unchanged
        rebuilt_any = rebuilt_any or rebuilt
        for stale_file_path, stale_count in stale_by_file:
            click.echo(f"[INFO] Deleted {stale_count} stale chunks for {stale_file_path}.")

    return total_inserted, total_updated, total_unchanged, total_chunks, rebuilt_any


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
    Ingest documents from file(s) or directory(ies) into SQLite (+ FAISS for dense/hybrid).

    Supported formats (best-effort): .txt, .md, .csv.
    """
    if not paths:
        click.echo("[ERROR] Provide one or more paths (file or directory).", err=True)
        raise SystemExit(2)

    try:
        from local_rag_backend.core.services.ingestion import build_preprocess_fn_from_settings
        from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
        from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

        doc_repo = SqlDocumentStorage()
        preprocess_fn = build_preprocess_fn_from_settings(settings)
        delimiter_opt, has_header = _resolve_loader_options(
            csv_delimiter=csv_delimiter,
            csv_has_header=csv_has_header,
        )

        embedder: EmbedderPort | None = None
        vec = None
        if settings.retrieval_mode in ("dense", "hybrid") and not dry_run:
            embedder = build_dense_embedder()
            vec = FaissVectorStorage(
                index_path=settings.index_path,
                id_map_path=settings.id_map_path,
                dim=embedder.dim,
            )

        files = _discover_input_files(
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

        ingest_plans, total_files, dry_run_chunks, total_skipped = _build_ingest_plans(
            files=files,
            preprocess_fn=preprocess_fn,
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

        def _ingest_sync() -> tuple[int, int, int, int, bool]:
            return _execute_ingest_batches(
                ingest_plans=ingest_plans,
                doc_repo=doc_repo,
                embedder=embedder,
                vec=vec,
            )

        total_inserted, total_updated, total_unchanged, total_chunks, rebuilt_any = (
            run_cli_mutation(_ingest_sync)
        )
        click.echo(
            f"[OK] Ingest completed. files={total_files} chunks={total_chunks} "
            f"inserted={total_inserted} updated={total_updated} unchanged={total_unchanged} "
            f"skipped={total_skipped} rebuilt_index={rebuilt_any}"
        )
    except Exception as e:
        click.echo(f"[ERROR] Error ingesting files: {e}", err=True)
        raise SystemExit(1)

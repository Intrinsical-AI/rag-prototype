# src/cli.py
"""
CLI entry points for Intrinsical RAG Prototype.

This module provides command-line interfaces for common operations like
starting the server, building indices, and bootstrapping data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import click
import uvicorn

from local_rag_backend import __version__
from local_rag_backend.app.diagnostics import (
    get_documents_count,
    get_history_count,
    get_retrieval_index_stats,
)
from local_rag_backend.core.services.dense_upsert import (
    precompute_vectors_for_changed_items,
    sync_dense_after_upsert,
)
from local_rag_backend.infrastructure.persistence.faiss.manifest import (
    expected_manifest_config_from_settings,
)
from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from local_rag_backend.core.ports import EmbedderPort

T = TypeVar("T")


def _ensure_sqlite_schema_for_cli() -> None:
    """
    Ensure SQLite schema is compatible with the current ORM mappings.

    CLI commands can be run without starting the FastAPI server, so they must
    apply the same best-effort SQLite migrations that the app does at startup.
    """
    from local_rag_backend.infrastructure.persistence.sqlalchemy import base as db_base

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db_base.Base.metadata.create_all(bind=db_base.engine)
    db_base.ensure_sqlite_documents_autoincrement(
        engine_to_use=db_base.engine, id_map_path=str(settings.id_map_path)
    )
    db_base.ensure_sqlite_documents_identity_columns(engine_to_use=db_base.engine)


@click.group()
@click.version_option(version=__version__, prog_name="rag-prototype")
def cli() -> None:
    """Intrinsical RAG Prototype - Production-ready RAG system with hexagonal architecture."""
    pass


def _run_with_multi_store_write_lock(operation: Callable[[], T]) -> T:
    from local_rag_backend.core.services.write_lock import multi_store_write_lock

    with multi_store_write_lock():
        return operation()


def _reset_rag_service_best_effort() -> None:
    from local_rag_backend.app.factory import reset_rag_service

    try:
        reset_rag_service()
    except Exception:
        return None


def _build_dense_embedder() -> EmbedderPort:
    """
    Build the dense/hybrid embedder based on current settings.

    Keep this in one place so CLI mutating commands (rebuild/upsert/ingest/delete)
    stay consistent when embedding backend selection changes.
    """
    from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder
    from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
        SentenceTransformerEmbedder,
    )

    return (
        OpenAIEmbedder()
        if settings.openai_api_key
        else SentenceTransformerEmbedder(model_name=settings.st_embedding_model)
    )


def _batched(values: list[T], batch_size: int) -> list[list[T]]:
    size = max(1, int(batch_size))
    return [values[i : i + size] for i in range(0, len(values), size)]


@cli.command()
def server() -> None:
    """Start the RAG FastAPI server using settings from config file or environment."""
    click.echo(f"🚀 Starting server on {settings.app_host}:{settings.app_port}...")
    click.echo(f"   - Mode: {'Development (reload)' if settings.debug else 'Production'}")
    click.echo(f"   - Log level: {settings.log_level.upper()}")
    click.echo(f"   - Retrieval: {settings.retrieval_mode}")

    uvicorn.run(
        "local_rag_backend.app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


@cli.command("build-index")
def build_index() -> None:
    """Build FAISS index from existing documents."""
    mutation_attempted = False
    try:
        _ensure_sqlite_schema_for_cli()
        # Import here to avoid circular imports
        from local_rag_backend.scripts.build_index import main as build_main

        click.echo("[INFO] Building FAISS index...")
        with click.progressbar(length=1, label="Building index") as bar:
            mutation_attempted = True
            _run_with_multi_store_write_lock(build_main)
            bar.update(1)
        click.echo("[OK] Index built successfully!")
    except Exception as e:
        click.echo(f"[ERROR] Error building index: {e}", err=True)
        sys.exit(1)
    finally:
        if mutation_attempted:
            _reset_rag_service_best_effort()


@cli.command("rebuild-index")
def rebuild_index() -> None:
    """Rebuild FAISS index from the current SQLite documents (idempotent)."""
    mutation_attempted = False
    try:
        _ensure_sqlite_schema_for_cli()
        from local_rag_backend.core.services.maintenance import rebuild_index_from_db
        from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
        from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

        if settings.retrieval_mode not in ("dense", "hybrid"):
            raise RuntimeError("rebuild-index requires RETRIEVAL_MODE=dense|hybrid")

        def _rebuild_sync() -> int:
            doc_repo = SqlDocumentStorage()
            embedder = _build_dense_embedder()
            # Rebuild must be able to recover from an incompatible on-disk index (e.g. dim drift).
            from local_rag_backend.infrastructure.persistence.faiss.manifest import (
                purge_index_artifacts,
            )

            purge_index_artifacts(index_path=settings.index_path, id_map_path=settings.id_map_path)
            vec = FaissVectorStorage(
                index_path=settings.index_path, id_map_path=settings.id_map_path, dim=embedder.dim
            )
            return rebuild_index_from_db(doc_repo=doc_repo, vec_repo=vec, embedder=embedder)

        mutation_attempted = True
        n = _run_with_multi_store_write_lock(_rebuild_sync)
        click.echo(f"[OK] Rebuilt index with {n} vectors.")
    except Exception as e:
        click.echo(f"[ERROR] Error rebuilding index: {e}", err=True)
        sys.exit(1)
    finally:
        if mutation_attempted:
            _reset_rag_service_best_effort()


@cli.command("delete-docs")
@click.argument("ids", nargs=-1, type=int)
def delete_docs(ids: tuple[int, ...]) -> None:
    """Delete documents by ID from SQLite (and FAISS in dense/hybrid mode)."""
    if not ids:
        click.echo("[ERROR] Provide one or more document IDs.", err=True)
        sys.exit(2)

    mutation_attempted = False
    try:
        _ensure_sqlite_schema_for_cli()
        from local_rag_backend.core.services.maintenance import delete_documents_multi_store
        from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
        from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

        def _delete_sync() -> tuple[int, int | None, bool]:
            doc_repo = SqlDocumentStorage()
            if settings.retrieval_mode in ("dense", "hybrid"):
                # For deletion, avoid loading the embedder just to infer dim if the index exists.
                vec = FaissVectorStorage(
                    index_path=settings.index_path, id_map_path=settings.id_map_path, dim=None
                )
                return delete_documents_multi_store(
                    doc_repo=doc_repo,
                    vec_repo=vec,
                    embedder_factory=_build_dense_embedder,
                    ids=list(ids),
                    rebuild_on_index_failure=True,
                )

            deleted_sql, _, rebuilt = delete_documents_multi_store(doc_repo=doc_repo, ids=list(ids))
            return deleted_sql, None, rebuilt

        mutation_attempted = True
        deleted_sql, deleted_index, rebuilt = _run_with_multi_store_write_lock(_delete_sync)
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
        sys.exit(1)
    finally:
        if mutation_attempted:
            _reset_rag_service_best_effort()


@cli.command("delete-external-ids")
@click.argument("external_ids", nargs=-1, type=str)
def delete_external_ids(external_ids: tuple[str, ...]) -> None:
    """Delete documents by external_id from SQLite (and FAISS in dense/hybrid mode), adding tombstones."""
    if not external_ids:
        click.echo("[ERROR] Provide one or more external_ids.", err=True)
        sys.exit(2)

    mutation_attempted = False
    try:
        _ensure_sqlite_schema_for_cli()
        from local_rag_backend.core.services.maintenance import rebuild_index_from_db
        from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
        from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

        def _delete_sync() -> tuple[int, int | None, list[str], int, bool]:
            doc_repo = SqlDocumentStorage()
            deleted_sql, deleted_ids, missing, tombstoned = doc_repo.delete_by_external_ids(
                list(external_ids)
            )

            rebuilt = False
            deleted_index: int | None = None
            if settings.retrieval_mode in ("dense", "hybrid"):
                vec = FaissVectorStorage(
                    index_path=settings.index_path,
                    id_map_path=settings.id_map_path,
                    dim=None,
                )
                try:
                    deleted_index = int(vec.delete(deleted_ids))
                except Exception:
                    n = rebuild_index_from_db(
                        doc_repo=doc_repo, vec_repo=vec, embedder=_build_dense_embedder()
                    )
                    rebuilt = n >= 0
            return deleted_sql, deleted_index, missing, tombstoned, rebuilt

        mutation_attempted = True
        deleted_sql, deleted_index, missing, tombstoned, rebuilt = _run_with_multi_store_write_lock(
            _delete_sync
        )
        click.echo(
            f"[OK] Deleted {deleted_sql} docs by external_id. "
            f"tombstoned={tombstoned} missing={len(missing)} "
            f"index_delete={'ok' if deleted_index is not None else 'n/a'} rebuilt={rebuilt}."
        )
        if missing:
            click.echo(f"[INFO] Missing external_ids (tombstoned anyway): {missing[:10]}")
    except Exception as e:
        click.echo(f"[ERROR] Error deleting by external_id: {e}", err=True)
        sys.exit(1)
    finally:
        if mutation_attempted:
            _reset_rag_service_best_effort()


@cli.command("upsert-docs")
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
def upsert_docs(
    json_path: Path | None,
    external_id: str | None,
    content: str | None,
    source_id: str | None,
    metadata_json: str | None,
) -> None:
    """Upsert documents by external_id (idempotent)."""
    mutation_attempted = False
    try:
        _ensure_sqlite_schema_for_cli()
        from typing import Any, cast

        from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
        from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

        docs_payload: list[dict[str, object]] = []
        if json_path is not None:
            docs_payload = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(docs_payload, list):
                raise ValueError("--json must contain a JSON list of documents")
        else:
            if not external_id or not content:
                raise ValueError(
                    "Provide --json or both --external-id and --content for a single document."
                )
            md_single = None
            if metadata_json:
                md_single = json.loads(metadata_json)
                if not isinstance(md_single, dict):
                    raise ValueError("--metadata-json must be a JSON object")
            docs_payload = [
                {
                    "external_id": external_id,
                    "content": content,
                    "source_id": source_id,
                    "metadata": md_single,
                }
            ]

        items = []
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

        def _upsert_sync() -> tuple[int, int, int, bool]:
            doc_repo = SqlDocumentStorage()
            tombstoned = doc_repo.get_tombstoned_external_ids([i.external_id for i in items])
            if tombstoned:
                raise RuntimeError(
                    "Some external_id values are tombstoned (deleted): "
                    + ", ".join(sorted(tombstoned)[:10])
                )
            embedder = None
            vectors_by_external_id: dict[str, list[float]] = {}
            if settings.retrieval_mode in ("dense", "hybrid"):
                embedder = _build_dense_embedder()
                vectors_by_external_id = precompute_vectors_for_changed_items(
                    items=items,
                    doc_repo=doc_repo,
                    embedder=embedder,
                )

            results, _changed_content, updated_content_ids = (
                doc_repo.upsert_documents_by_external_id(items)
            )

            inserted = sum(1 for r in results if r.action == "inserted")
            updated = sum(1 for r in results if r.action == "updated")
            unchanged = sum(1 for r in results if r.action == "unchanged")

            rebuilt = False
            if settings.retrieval_mode in ("dense", "hybrid") and embedder is not None:
                vec = FaissVectorStorage(
                    index_path=settings.index_path,
                    id_map_path=settings.id_map_path,
                    dim=embedder.dim,
                )
                rebuilt = sync_dense_after_upsert(
                    results=results,
                    updated_content_ids=updated_content_ids,
                    vectors_by_external_id=vectors_by_external_id,
                    vec_repo=vec,
                    doc_repo=doc_repo,
                    embedder=embedder,
                )

            return inserted, updated, unchanged, rebuilt

        mutation_attempted = True
        inserted, updated, unchanged, rebuilt = _run_with_multi_store_write_lock(_upsert_sync)
        click.echo(
            f"[OK] Upserted docs. inserted={inserted} updated={updated} unchanged={unchanged} rebuilt_index={rebuilt}"
        )
    except Exception as e:
        click.echo(f"[ERROR] Error upserting docs: {e}", err=True)
        sys.exit(1)
    finally:
        if mutation_attempted:
            _reset_rag_service_best_effort()


@cli.command()
def bootstrap() -> None:
    """Bootstrap database with sample data."""
    mutation_attempted = False
    try:
        _ensure_sqlite_schema_for_cli()
        # Import here to avoid circular imports
        from local_rag_backend.scripts.bootstrap import main as bootstrap_main

        click.echo("[INFO] Bootstrapping database with sample data...")
        with click.progressbar(length=1, label="Bootstrapping") as bar:
            mutation_attempted = True
            _run_with_multi_store_write_lock(bootstrap_main)
            bar.update(1)
        click.echo("[OK] Bootstrap completed successfully!")
    except Exception as e:
        click.echo(f"[ERROR] Error bootstrapping: {e}", err=True)
        sys.exit(1)
    finally:
        if mutation_attempted:
            _reset_rag_service_best_effort()


@cli.command()
def status() -> None:
    """Display system status and configuration."""
    from local_rag_backend.infrastructure.persistence.sqlalchemy import base as db_base

    _ensure_sqlite_schema_for_cli()

    # Styles
    title_fg = "cyan"
    key_fg = "blue"

    click.secho("🔍 Intrinsical RAG Prototype - System Status", fg=title_fg, bold=True)
    click.echo()

    # Core config
    click.secho("📋 Core Configuration", fg=title_fg, bold=True)
    click.echo(
        f"  {click.style('Host:', fg=key_fg, bold=True)} {settings.app_host}:{settings.app_port}"
    )
    click.echo(
        f"  {click.style('Retrieval Mode:', fg=key_fg, bold=True)} {settings.retrieval_mode}"
    )
    click.echo(
        f"  {click.style('Debug Mode:', fg=key_fg, bold=True)} {'✅' if settings.debug else '❌'}"
    )
    click.echo()

    # LLM config
    click.secho("🤖 LLM Configuration", fg=title_fg, bold=True)
    click.echo(
        f"  {click.style('Ollama Enabled:', fg=key_fg, bold=True)} {'✅' if settings.ollama_enabled else '❌'}"
    )
    if settings.ollama_enabled:
        click.echo(f"    {click.style('URL:', fg=key_fg, bold=True)} {settings.ollama_base_url}")
        click.echo(f"    {click.style('Model:', fg=key_fg, bold=True)} {settings.ollama_model}")
    click.echo(
        f"  {click.style('OpenAI Enabled:', fg=key_fg, bold=True)} {'✅' if bool(settings.openai_api_key) else '❌'}"
    )
    if settings.openai_api_key:
        click.echo(f"    {click.style('Model:', fg=key_fg, bold=True)} {settings.openai_model}")
    click.echo()

    # File status
    click.secho("📁 File Status", fg=title_fg, bold=True)
    for name, path_str in [
        ("Database", settings.sqlite_url.replace("sqlite:///", "")),
        ("FAISS index", settings.index_path),
        ("Sample data", settings.faq_csv),
    ]:
        path = Path(path_str)
        status_icon = "✅ YES" if path.exists() else "❌ NO"
        click.echo(f"  {click.style(name + ':', fg=key_fg, bold=True)} {status_icon}")
        click.echo(f"    Path: {path}")

    # Consistency / counts (best-effort)
    click.echo()
    click.secho("📊 Data & Index Diagnostics", fg=title_fg, bold=True)

    docs_count: int | None = None
    try:
        docs_count = get_documents_count(db_base.engine)
        click.echo(f"  {click.style('Documents:', fg=key_fg, bold=True)} {docs_count}")
    except Exception as e:
        click.echo(f"  {click.style('Documents:', fg=key_fg, bold=True)} [ERROR] {e!s}")

    try:
        hist = get_history_count(db_base.engine)
        click.echo(f"  {click.style('History:', fg=key_fg, bold=True)} {hist}")
    except Exception as e:
        click.echo(f"  {click.style('History:', fg=key_fg, bold=True)} [WARN] {e!s}")

    if settings.retrieval_mode in ("dense", "hybrid"):
        expected_manifest = expected_manifest_config_from_settings(settings)
        stats = get_retrieval_index_stats(
            index_path=settings.index_path,
            id_map_path=settings.id_map_path,
            dim=None,
            expected_manifest=expected_manifest,
        )
        status_txt = str(stats.get("status"))
        if status_txt == "ok":
            click.echo(
                f"  {click.style('Index:', fg=key_fg, bold=True)} "
                f"{stats.get('vectors')} vectors "
                f"(dim={stats.get('dim')}, backend={stats.get('backend')})"
            )
            click.echo(
                f"  {click.style('Manifest:', fg=key_fg, bold=True)} OK "
                f"(path={stats.get('manifest_path')})"
            )
            if int(stats.get("duplicates") or 0):
                click.echo(
                    f"  {click.style('Index drift:', fg=key_fg, bold=True)} "
                    f"[ERROR] duplicates={stats.get('duplicates')} "
                    f"(hint: {stats.get('hint')})"
                )
            elif docs_count is not None and int(stats.get("id_map_len") or 0) != docs_count:
                click.echo(
                    f"  {click.style('Index drift:', fg=key_fg, bold=True)} "
                    f"[ERROR] documents={docs_count} id_map={stats.get('id_map_len')} "
                    "(hint: run `rag-rebuild-index`)"
                )
            else:
                click.echo(f"  {click.style('Index drift:', fg=key_fg, bold=True)} OK")
        else:
            click.echo(
                f"  {click.style('Index:', fg=key_fg, bold=True)} "
                f"[ERROR] {status_txt} "
                f"(hint: {stats.get('hint')})"
            )
            if status_txt == "drift":
                mm = stats.get("manifest_mismatches") or []
                if isinstance(mm, list) and mm:
                    # Keep output short but actionable.
                    sample = ", ".join(
                        f"{m.get('key')}={m.get('actual')} (expected {m.get('expected')})"
                        for m in mm[:3]
                        if isinstance(m, dict)
                    )
                    click.echo(
                        f"  {click.style('Manifest drift:', fg=key_fg, bold=True)} [ERROR] {sample}"
                    )


@cli.command("eval")
@click.option(
    "--dataset",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a JSONL eval dataset. If omitted, uses the packaged rag_eval_v1 dataset.",
)
@click.option(
    "--retrieval-mode",
    type=click.Choice(["sparse"], case_sensitive=False),
    default="sparse",
    show_default=True,
)
@click.option("--k", type=int, default=3, show_default=True)
@click.option("--max-queries", type=int, default=None, help="Evaluate only the first N queries.")
@click.option("--reranker/--no-reranker", default=False, show_default=True)
@click.option("--fail-below-hit-rate", type=float, default=1.0, show_default=True)
@click.option("--fail-below-mrr", type=float, default=0.9, show_default=True)
@click.option(
    "--json-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to write eval results as JSON.",
)
def eval_cmd(
    dataset: Path | None,
    retrieval_mode: str,
    k: int,
    max_queries: int | None,
    reranker: bool,
    fail_below_hit_rate: float,
    fail_below_mrr: float,
    json_out: Path | None,
) -> None:
    """Offline retrieval evaluation (reproducible, dependency-free by default)."""
    from local_rag_backend.core.services.evaluation import (
        eval_result_to_json,
        format_eval_result,
        load_eval_dataset,
        run_retrieval_eval,
    )

    ds = load_eval_dataset(dataset)
    res = run_retrieval_eval(
        dataset=ds,
        retrieval_mode=retrieval_mode,
        k=k,
        reranker_enabled=bool(reranker),
        reranker_candidate_k=settings.reranker_candidate_k,
        reranker_strategy=settings.reranker_strategy,
        max_queries=max_queries,
    )
    click.echo(format_eval_result(res))

    if json_out is not None:
        json_out.write_text(json.dumps(eval_result_to_json(res)), encoding="utf-8")

    if res.hit_rate < float(fail_below_hit_rate) or res.mrr < float(fail_below_mrr):
        click.echo(
            f"[ERROR] Eval regression: hit_rate={res.hit_rate:.3f} (min {fail_below_hit_rate}), "
            f"mrr={res.mrr:.3f} (min {fail_below_mrr})",
            err=True,
        )
        sys.exit(1)


# Entry point functions for setuptools
def rag_server() -> None:
    """Entry point for rag-server command."""
    cli.main(args=["server"], standalone_mode=False)


def rag_build_index() -> None:
    """Entry point for rag-build-index command."""
    cli.main(args=["build-index", *sys.argv[1:]], standalone_mode=False)


def rag_bootstrap() -> None:
    """Entry point for rag-bootstrap command."""
    cli.main(args=["bootstrap", *sys.argv[1:]], standalone_mode=False)


def rag_status() -> None:
    """Entry point for rag-status command."""
    cli.main(args=["status", *sys.argv[1:]], standalone_mode=False)


def rag_eval() -> None:
    """Entry point for rag-eval command."""
    cli.main(args=["eval", *sys.argv[1:]], standalone_mode=False)


def rag_rebuild_index() -> None:
    """Entry point for rag-rebuild-index command."""
    cli.main(args=["rebuild-index", *sys.argv[1:]], standalone_mode=False)


def rag_delete_docs() -> None:
    """Entry point for rag-delete-docs command."""
    cli.main(args=["delete-docs", *sys.argv[1:]], standalone_mode=False)


def rag_upsert_docs() -> None:
    """Entry point for rag-upsert-docs command."""
    cli.main(args=["upsert-docs", *sys.argv[1:]], standalone_mode=False)


def rag_delete_external_ids() -> None:
    """Entry point for rag-delete-external-ids command."""
    cli.main(args=["delete-external-ids", *sys.argv[1:]], standalone_mode=False)


@cli.command("ingest")
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
def ingest(
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
        sys.exit(2)

    mutation_attempted = False
    try:
        _ensure_sqlite_schema_for_cli()
        from local_rag_backend.core.services.chunking import chunk_chars_v1
        from local_rag_backend.core.services.ingestion import (
            build_preprocess_fn_from_settings,
            default_formatter,
        )
        from local_rag_backend.core.services.maintenance import delete_documents_multi_store
        from local_rag_backend.infrastructure.ingestion.loaders.discovery import discover_files
        from local_rag_backend.infrastructure.ingestion.loaders.factory import (
            detect_file_format,
            get_loader_for_file,
        )
        from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
        from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

        doc_repo = SqlDocumentStorage()
        preprocess_fn = build_preprocess_fn_from_settings(settings)

        delimiter_opt = None if csv_delimiter.strip().lower() == "auto" else csv_delimiter
        has_header = settings.csv_has_header if csv_has_header is None else bool(csv_has_header)

        embedder: EmbedderPort | None = None
        vec = None
        if settings.retrieval_mode in ("dense", "hybrid") and not dry_run:
            embedder = _build_dense_embedder()
            vec = FaissVectorStorage(
                index_path=settings.index_path, id_map_path=settings.id_map_path, dim=embedder.dim
            )

        inputs = list(paths)
        files = list(
            discover_files(
                inputs,
                recursive=recursive,
                follow_symlinks=follow_symlinks,
                max_files=max_files,
                max_file_bytes=max_file_bytes,
                max_total_bytes=max_total_bytes,
            )
        )
        if not files:
            click.echo("[WARN] No files found under limits. Nothing to ingest.")
            return

        if not dry_run:
            mutation_attempted = True

        total_inserted = 0
        total_updated = 0
        total_unchanged = 0
        total_chunks = 0
        total_files = 0
        total_skipped = 0
        rebuilt_any = False
        # Keep batch size conservative to reduce commit overhead without ballooning memory.
        ingest_batch_files = 64
        ingest_plans: list[
            tuple[
                Path,
                str,
                tuple[SqlDocumentStorage.UpsertDoc, ...],
                tuple[str, ...],
            ]
        ] = []

        for file_path in files:
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
                total_skipped += 1
                continue

            source_id = str(file_path.resolve())
            # Include a delimiter to avoid accidental prefix matches:
            # e.g. `file:/tmp/foo:` should not match `file:/tmp/foo2:...`.
            file_prefix = f"file:{source_id}:"

            items: list[SqlDocumentStorage.UpsertDoc] = []
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
                total_skipped += 1
                continue

            if dry_run:
                total_files += 1
                total_chunks += len(items)
                continue

            total_files += 1
            ingest_plans.append((file_path, file_prefix, tuple(items), tuple(desired_external_ids)))

        if dry_run:
            click.echo(
                f"[DRY-RUN] files={total_files} chunks={total_chunks} skipped={total_skipped} "
                f"(limits: max_files={max_files} max_file_bytes={max_file_bytes} max_total_bytes={max_total_bytes})"
            )
            return

        for plan_batch in _batched(ingest_plans, ingest_batch_files):

            def _ingest_batch_sync(
                plans_bound: tuple[
                    tuple[
                        Path,
                        str,
                        tuple[SqlDocumentStorage.UpsertDoc, ...],
                        tuple[str, ...],
                    ],
                    ...,
                ] = tuple(plan_batch),
            ) -> tuple[int, int, int, bool, int, int, list[tuple[Path, int]]]:
                all_items: list[SqlDocumentStorage.UpsertDoc] = []
                stale_ids_all: list[int] = []
                stale_by_file: list[tuple[Path, int]] = []
                ingested_chunks = 0

                for file_path_bound, file_prefix_bound, items_bound, desired_bound in plans_bound:
                    desired_external_ids_local = set(desired_bound)
                    items_local = list(items_bound)

                    # Respect tombstones: deleted external_ids must not reappear on future ingestions.
                    tombstoned = doc_repo.get_tombstoned_external_ids(
                        list(desired_external_ids_local)
                    )
                    if tombstoned:
                        desired_external_ids_local -= tombstoned
                        items_local = [it for it in items_local if it.external_id not in tombstoned]

                    existing = doc_repo.list_ids_by_external_id_prefix(file_prefix_bound)
                    stale_ids = [
                        doc_id for doc_id, ext in existing if ext not in desired_external_ids_local
                    ]
                    if stale_ids:
                        stale_ids_all.extend(stale_ids)
                        stale_by_file.append((file_path_bound, len(stale_ids)))

                    ingested_chunks += len(items_local)
                    all_items.extend(items_local)

                # Defensive dedup: avoid duplicate external_ids when inputs overlap.
                unique_items: list[SqlDocumentStorage.UpsertDoc] = []
                seen_external_ids: set[str] = set()
                for item in all_items:
                    if item.external_id in seen_external_ids:
                        continue
                    seen_external_ids.add(item.external_id)
                    unique_items.append(item)

                vectors_by_external_id: dict[str, list[float]] = {}
                if settings.retrieval_mode in ("dense", "hybrid") and unique_items:
                    assert embedder is not None
                    vectors_by_external_id = precompute_vectors_for_changed_items(
                        items=unique_items,
                        doc_repo=doc_repo,
                        embedder=embedder,
                    )

                results: list[SqlDocumentStorage.UpsertResult] = []
                updated_content_ids: list[int] = []
                if unique_items:
                    results, _changed_content, updated_content_ids = (
                        doc_repo.upsert_documents_by_external_id(unique_items)
                    )

                inserted = sum(1 for r in results if r.action == "inserted")
                updated = sum(1 for r in results if r.action == "updated")
                unchanged = sum(1 for r in results if r.action == "unchanged")
                rebuilt = False
                deleted_stale = 0
                stale_ids_unique = sorted({int(x) for x in stale_ids_all})

                if settings.retrieval_mode in ("dense", "hybrid"):
                    assert embedder is not None
                    assert vec is not None

                    rebuilt = sync_dense_after_upsert(
                        results=results,
                        updated_content_ids=updated_content_ids,
                        vectors_by_external_id=vectors_by_external_id,
                        vec_repo=vec,
                        doc_repo=doc_repo,
                        embedder=embedder,
                    )

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
                elif stale_ids_unique:
                    deleted_sql, _, _ = delete_documents_multi_store(
                        doc_repo=doc_repo, ids=stale_ids_unique
                    )
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

            (
                inserted,
                updated,
                unchanged,
                rebuilt,
                _deleted_stale,
                ingested_chunks,
                stale_by_file,
            ) = _run_with_multi_store_write_lock(_ingest_batch_sync)

            total_chunks += ingested_chunks
            total_inserted += inserted
            total_updated += updated
            total_unchanged += unchanged
            rebuilt_any = rebuilt_any or rebuilt
            for stale_file_path, stale_count in stale_by_file:
                click.echo(f"[INFO] Deleted {stale_count} stale chunks for {stale_file_path}.")

        click.echo(
            f"[OK] Ingest completed. files={total_files} chunks={total_chunks} "
            f"inserted={total_inserted} updated={total_updated} unchanged={total_unchanged} "
            f"skipped={total_skipped} rebuilt_index={rebuilt_any}"
        )
    except Exception as e:
        click.echo(f"[ERROR] Error ingesting files: {e}", err=True)
        sys.exit(1)
    finally:
        if mutation_attempted:
            _reset_rag_service_best_effort()


def rag_ingest() -> None:
    """Entry point for rag-ingest command."""
    cli.main(args=["ingest", *sys.argv[1:]], standalone_mode=False)


if __name__ == "__main__":
    cli()

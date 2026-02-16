# src/cli.py
"""
CLI entry points for Intrinsical RAG Prototype.

This module provides command-line interfaces for common operations like
starting the server, building indices, and bootstrapping data.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import click
import uvicorn

from local_rag_backend import __version__
from local_rag_backend.app.diagnostics import (
    get_documents_count,
    get_history_count,
    get_retrieval_index_stats,
)
from local_rag_backend.settings import settings


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
    try:
        _ensure_sqlite_schema_for_cli()
        # Import here to avoid circular imports
        from local_rag_backend.scripts.build_index import main as build_main

        click.echo("[INFO] Building FAISS index...")
        with click.progressbar(length=1, label="Building index") as bar:
            build_main()
            bar.update(1)
        click.echo("[OK] Index built successfully!")
    except Exception as e:
        click.echo(f"[ERROR] Error building index: {e}", err=True)
        sys.exit(1)


@cli.command("rebuild-index")
def rebuild_index() -> None:
    """Rebuild FAISS index from the current SQLite documents (idempotent)."""
    try:
        _ensure_sqlite_schema_for_cli()
        from local_rag_backend.app.factory import reset_rag_service
        from local_rag_backend.core.services.maintenance import rebuild_index_from_db
        from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder
        from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
            SentenceTransformerEmbedder,
        )
        from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
        from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

        if settings.retrieval_mode not in ("dense", "hybrid"):
            raise RuntimeError("rebuild-index requires RETRIEVAL_MODE=dense|hybrid")

        doc_repo = SqlDocumentStorage()
        embedder = (
            OpenAIEmbedder()
            if settings.openai_api_key
            else SentenceTransformerEmbedder(model_name=settings.st_embedding_model)
        )
        # Rebuild must be able to recover from an incompatible on-disk index (e.g. dim drift).
        from local_rag_backend.infrastructure.persistence.faiss.manifest import (
            purge_index_artifacts,
        )

        purge_index_artifacts(index_path=settings.index_path, id_map_path=settings.id_map_path)
        vec = FaissVectorStorage(
            index_path=settings.index_path, id_map_path=settings.id_map_path, dim=embedder.dim
        )
        n = rebuild_index_from_db(doc_repo=doc_repo, vec_repo=vec, embedder=embedder)
        reset_rag_service()
        click.echo(f"[OK] Rebuilt index with {n} vectors.")
    except Exception as e:
        click.echo(f"[ERROR] Error rebuilding index: {e}", err=True)
        sys.exit(1)


@cli.command("delete-docs")
@click.argument("ids", nargs=-1, type=int)
def delete_docs(ids: tuple[int, ...]) -> None:
    """Delete documents by ID from SQLite (and FAISS in dense/hybrid mode)."""
    if not ids:
        click.echo("[ERROR] Provide one or more document IDs.", err=True)
        sys.exit(2)

    try:
        _ensure_sqlite_schema_for_cli()
        from local_rag_backend.app.factory import reset_rag_service
        from local_rag_backend.core.services.maintenance import delete_documents_multi_store
        from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder
        from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
            SentenceTransformerEmbedder,
        )
        from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
        from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

        doc_repo = SqlDocumentStorage()
        if settings.retrieval_mode in ("dense", "hybrid"):
            # For deletion, avoid loading the embedder just to infer dim if the index already exists.
            vec = FaissVectorStorage(
                index_path=settings.index_path, id_map_path=settings.id_map_path, dim=None
            )
            embedder = (
                OpenAIEmbedder()
                if settings.openai_api_key
                else SentenceTransformerEmbedder(model_name=settings.st_embedding_model)
            )
            deleted_sql, deleted_index, rebuilt = delete_documents_multi_store(
                doc_repo=doc_repo,
                vec_repo=vec,
                embedder=embedder,
                ids=list(ids),
                rebuild_on_index_failure=True,
            )
            reset_rag_service()
            click.echo(
                f"[OK] Deleted {deleted_sql} docs from SQL. "
                f"Index delete={'ok' if deleted_index is not None else 'n/a'} "
                f"rebuilt={rebuilt}."
            )
            return

        deleted_sql, _, _ = delete_documents_multi_store(doc_repo=doc_repo, ids=list(ids))
        reset_rag_service()
        click.echo(f"[OK] Deleted {deleted_sql} docs from SQL.")
    except Exception as e:
        click.echo(f"[ERROR] Error deleting docs: {e}", err=True)
        sys.exit(1)


@cli.command("delete-external-ids")
@click.argument("external_ids", nargs=-1, type=str)
def delete_external_ids(external_ids: tuple[str, ...]) -> None:
    """Delete documents by external_id from SQLite (and FAISS in dense/hybrid mode), adding tombstones."""
    if not external_ids:
        click.echo("[ERROR] Provide one or more external_ids.", err=True)
        sys.exit(2)

    try:
        _ensure_sqlite_schema_for_cli()
        from local_rag_backend.app.factory import reset_rag_service
        from local_rag_backend.core.services.maintenance import rebuild_index_from_db
        from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder
        from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
            SentenceTransformerEmbedder,
        )
        from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
        from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

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
            embedder = (
                OpenAIEmbedder()
                if settings.openai_api_key
                else SentenceTransformerEmbedder(model_name=settings.st_embedding_model)
            )
            try:
                vec.delete(deleted_ids)
                deleted_index = len(deleted_ids)
            except Exception:
                n = rebuild_index_from_db(doc_repo=doc_repo, vec_repo=vec, embedder=embedder)
                rebuilt = n >= 0

        reset_rag_service()
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
    try:
        _ensure_sqlite_schema_for_cli()
        from typing import Any, cast

        from local_rag_backend.app.factory import reset_rag_service
        from local_rag_backend.core.services.maintenance import rebuild_index_from_db
        from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder
        from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
            SentenceTransformerEmbedder,
        )
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
            embedder = (
                OpenAIEmbedder()
                if settings.openai_api_key
                else SentenceTransformerEmbedder(model_name=settings.st_embedding_model)
            )
            existing_states = doc_repo.get_existing_doc_states_by_external_id(
                [it.external_id for it in items]
            )
            to_embed = []
            for it in items:
                content = it.content.strip()
                current = existing_states.get(it.external_id)
                if current is None:
                    to_embed.append(it)
                    continue
                content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
                old_sha = current.content_sha256 or ""
                if old_sha != content_sha or current.content != content:
                    to_embed.append(it)

            if to_embed:
                embedded = embedder.embed([it.content.strip() for it in to_embed])
                if len(embedded) != len(to_embed):
                    raise RuntimeError(
                        f"Embedder returned {len(embedded)} vectors for {len(to_embed)} documents."
                    )
                vectors_by_external_id = {
                    it.external_id: list(vec) for it, vec in zip(to_embed, embedded, strict=False)
                }

        results, _changed_content, updated_content_ids = doc_repo.upsert_documents_by_external_id(
            items
        )

        inserted = sum(1 for r in results if r.action == "inserted")
        updated = sum(1 for r in results if r.action == "updated")
        unchanged = sum(1 for r in results if r.action == "unchanged")

        rebuilt = False
        if settings.retrieval_mode in ("dense", "hybrid") and embedder is not None:
            changed_results = [r for r in results if r.content_changed]
            missing_vectors = [
                r.external_id
                for r in changed_results
                if r.external_id not in vectors_by_external_id
            ]
            if missing_vectors:
                raise RuntimeError(
                    "Missing precomputed vectors for changed documents: "
                    + ", ".join(missing_vectors[:10])
                )
            ids = [int(r.id) for r in changed_results]
            vectors = [vectors_by_external_id[r.external_id] for r in changed_results]
            if not ids:
                reset_rag_service()
                click.echo(
                    f"[OK] Upserted docs. inserted={inserted} updated={updated} unchanged={unchanged} rebuilt_index={rebuilt}"
                )
                return
            vec = FaissVectorStorage(
                index_path=settings.index_path,
                id_map_path=settings.id_map_path,
                dim=embedder.dim,
            )
            try:
                if updated_content_ids:
                    vec.delete(updated_content_ids)
                vec.upsert(ids, vectors)
            except Exception:
                n = rebuild_index_from_db(doc_repo=doc_repo, vec_repo=vec, embedder=embedder)
                rebuilt = n >= 0

        reset_rag_service()
        click.echo(
            f"[OK] Upserted docs. inserted={inserted} updated={updated} unchanged={unchanged} rebuilt_index={rebuilt}"
        )
    except Exception as e:
        click.echo(f"[ERROR] Error upserting docs: {e}", err=True)
        sys.exit(1)


@cli.command()
def bootstrap() -> None:
    """Bootstrap database with sample data."""
    try:
        _ensure_sqlite_schema_for_cli()
        # Import here to avoid circular imports
        from local_rag_backend.scripts.bootstrap import main as bootstrap_main

        click.echo("[INFO] Bootstrapping database with sample data...")
        with click.progressbar(length=1, label="Bootstrapping") as bar:
            bootstrap_main()
            bar.update(1)
        click.echo("[OK] Bootstrap completed successfully!")
    except Exception as e:
        click.echo(f"[ERROR] Error bootstrapping: {e}", err=True)
        sys.exit(1)


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
        expected_manifest = {
            "embedding_backend": (
                "openai" if bool(settings.openai_api_key) else "sentence_transformers"
            ),
            "embedding_model": (
                settings.openai_embedding_model
                if bool(settings.openai_api_key)
                else settings.st_embedding_model
            ),
            "chunker_strategy": settings.ingest_chunk_strategy,
            "chunker_version": settings.ingest_chunker_version,
        }
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

    try:
        _ensure_sqlite_schema_for_cli()
        from local_rag_backend.app.factory import reset_rag_service
        from local_rag_backend.core.services.chunking import chunk_chars_v1
        from local_rag_backend.core.services.ingestion import (
            build_preprocess_fn_from_settings,
            default_formatter,
        )
        from local_rag_backend.core.services.maintenance import (
            delete_documents_multi_store,
            rebuild_index_from_db,
        )
        from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder
        from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
            SentenceTransformerEmbedder,
        )
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

        embedder = None
        vec = None
        if settings.retrieval_mode in ("dense", "hybrid") and not dry_run:
            embedder = (
                OpenAIEmbedder()
                if settings.openai_api_key
                else SentenceTransformerEmbedder(model_name=settings.st_embedding_model)
            )
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

        total_inserted = 0
        total_updated = 0
        total_unchanged = 0
        total_chunks = 0
        total_files = 0
        total_skipped = 0
        rebuilt_any = False

        for file_path in files:
            det = detect_file_format(file_path, sniff_bytes=sniff_bytes, use_magic=use_magic)
            loader = get_loader_for_file(
                file_path,
                sniff_bytes=sniff_bytes,
                use_magic=use_magic,
                csv_delimiter=delimiter_opt,
                csv_has_header=has_header,
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

            # Respect tombstones: deleted external_ids must not reappear on future ingestions.
            tombstoned = doc_repo.get_tombstoned_external_ids(list(desired_external_ids))
            if tombstoned:
                desired_external_ids -= tombstoned
                items = [it for it in items if it.external_id not in tombstoned]

            existing = doc_repo.list_ids_by_external_id_prefix(file_prefix)
            stale_ids = [doc_id for doc_id, ext in existing if ext not in desired_external_ids]

            if dry_run:
                total_files += 1
                total_chunks += len(items)
                continue

            vectors_by_external_id: dict[str, list[float]] = {}
            if settings.retrieval_mode in ("dense", "hybrid"):
                assert embedder is not None
                existing_states = doc_repo.get_existing_doc_states_by_external_id(
                    [it.external_id for it in items]
                )
                to_embed = []
                for it in items:
                    content = it.content.strip()
                    current = existing_states.get(it.external_id)
                    if current is None:
                        to_embed.append(it)
                        continue
                    content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    old_sha = current.content_sha256 or ""
                    if old_sha != content_sha or current.content != content:
                        to_embed.append(it)

                if to_embed:
                    embedded = embedder.embed([it.content.strip() for it in to_embed])
                    if len(embedded) != len(to_embed):
                        raise RuntimeError(
                            f"Embedder returned {len(embedded)} vectors for {len(to_embed)} documents."
                        )
                    vectors_by_external_id = {
                        it.external_id: list(vec)
                        for it, vec in zip(to_embed, embedded, strict=False)
                    }

            results, _changed_content, updated_content_ids = (
                doc_repo.upsert_documents_by_external_id(items)
            )

            total_files += 1
            total_chunks += len(items)
            total_inserted += sum(1 for r in results if r.action == "inserted")
            total_updated += sum(1 for r in results if r.action == "updated")
            total_unchanged += sum(1 for r in results if r.action == "unchanged")

            if settings.retrieval_mode in ("dense", "hybrid"):
                assert embedder is not None
                assert vec is not None

                rebuilt = False
                changed_results = [r for r in results if r.content_changed]
                if changed_results:
                    missing_vectors = [
                        r.external_id
                        for r in changed_results
                        if r.external_id not in vectors_by_external_id
                    ]
                    if missing_vectors:
                        raise RuntimeError(
                            "Missing precomputed vectors for changed documents: "
                            + ", ".join(missing_vectors[:10])
                        )
                    ids = [int(r.id) for r in changed_results]
                    vectors = [vectors_by_external_id[r.external_id] for r in changed_results]
                    try:
                        if updated_content_ids:
                            vec.delete(updated_content_ids)
                        vec.upsert(ids, vectors)
                    except Exception:
                        n = rebuild_index_from_db(
                            doc_repo=doc_repo, vec_repo=vec, embedder=embedder
                        )
                        rebuilt = n >= 0

                if stale_ids:
                    deleted_sql, _, rebuilt_del = delete_documents_multi_store(
                        doc_repo=doc_repo,
                        ids=stale_ids,
                        vec_repo=vec,
                        embedder=embedder,
                        rebuild_on_index_failure=True,
                    )
                    total_deleted = deleted_sql
                    if total_deleted:
                        click.echo(f"[INFO] Deleted {total_deleted} stale chunks for {file_path}.")
                    rebuilt = rebuilt or rebuilt_del

                rebuilt_any = rebuilt_any or rebuilt
            else:
                if stale_ids:
                    deleted_sql, _, _ = delete_documents_multi_store(
                        doc_repo=doc_repo, ids=stale_ids
                    )
                    if deleted_sql:
                        click.echo(f"[INFO] Deleted {deleted_sql} stale chunks for {file_path}.")

        if not dry_run:
            reset_rag_service()

        if dry_run:
            click.echo(
                f"[DRY-RUN] files={total_files} chunks={total_chunks} skipped={total_skipped} "
                f"(limits: max_files={max_files} max_file_bytes={max_file_bytes} max_total_bytes={max_total_bytes})"
            )
            return

        click.echo(
            f"[OK] Ingest completed. files={total_files} chunks={total_chunks} "
            f"inserted={total_inserted} updated={total_updated} unchanged={total_unchanged} "
            f"skipped={total_skipped} rebuilt_index={rebuilt_any}"
        )
    except Exception as e:
        click.echo(f"[ERROR] Error ingesting files: {e}", err=True)
        sys.exit(1)


def rag_ingest() -> None:
    """Entry point for rag-ingest command."""
    cli.main(args=["ingest", *sys.argv[1:]], standalone_mode=False)


if __name__ == "__main__":
    cli()

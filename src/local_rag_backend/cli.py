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

import click
import uvicorn

from local_rag_backend import __version__
from local_rag_backend.diagnostics import (
    get_documents_count,
    get_history_count,
    get_retrieval_index_stats,
)
from local_rag_backend.settings import settings


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
        results, changed_content, updated_content_ids = doc_repo.upsert_documents_by_external_id(
            items
        )

        inserted = sum(1 for r in results if r.action == "inserted")
        updated = sum(1 for r in results if r.action == "updated")
        unchanged = sum(1 for r in results if r.action == "unchanged")

        rebuilt = False
        if settings.retrieval_mode in ("dense", "hybrid") and changed_content:
            embedder = (
                OpenAIEmbedder()
                if settings.openai_api_key
                else SentenceTransformerEmbedder(model_name=settings.st_embedding_model)
            )
            vec = FaissVectorStorage(
                index_path=settings.index_path,
                id_map_path=settings.id_map_path,
                dim=embedder.dim,
            )
            ids = [doc_id for doc_id, _ in changed_content]
            texts = [text for _, text in changed_content]
            vectors = embedder.embed(texts)
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
        stats = get_retrieval_index_stats(
            index_path=settings.index_path, id_map_path=settings.id_map_path, dim=None
        )
        status_txt = str(stats.get("status"))
        if status_txt == "ok":
            click.echo(
                f"  {click.style('Index:', fg=key_fg, bold=True)} "
                f"{stats.get('vectors')} vectors "
                f"(dim={stats.get('dim')}, backend={stats.get('backend')})"
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


if __name__ == "__main__":
    cli()

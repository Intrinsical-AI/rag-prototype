from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from local_rag_backend.app.diagnostics import (
    get_documents_count,
    get_history_count,
    get_retrieval_index_stats,
)
from local_rag_backend.infrastructure.persistence.faiss.manifest import (
    expected_manifest_config_from_settings,
)
from local_rag_backend.settings import settings


def _hooks() -> Any:
    from local_rag_backend import cli as cli_module

    return cli_module


@click.command("build-index")
def build_index_cmd() -> None:
    """Build FAISS index from existing documents."""
    mutation_attempted = False
    try:
        hooks = _hooks()
        hooks._ensure_sqlite_schema_for_cli()
        from local_rag_backend.scripts.build_index import main as build_main

        click.echo("[INFO] Building FAISS index...")
        with click.progressbar(length=1, label="Building index") as bar:
            mutation_attempted = True
            hooks._run_with_multi_store_write_lock(build_main)
            bar.update(1)
        click.echo("[OK] Index built successfully!")
    except Exception as e:
        click.echo(f"[ERROR] Error building index: {e}", err=True)
        raise SystemExit(1)
    finally:
        if mutation_attempted:
            _hooks()._reset_rag_service_best_effort()


@click.command("rebuild-index")
def rebuild_index_cmd() -> None:
    """Rebuild FAISS index from the current SQLite documents (idempotent)."""
    mutation_attempted = False
    try:
        hooks = _hooks()
        hooks._ensure_sqlite_schema_for_cli()
        from local_rag_backend.core.services.maintenance import rebuild_index_from_db
        from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
        from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

        if settings.retrieval_mode not in ("dense", "hybrid"):
            raise RuntimeError("rebuild-index requires RETRIEVAL_MODE=dense|hybrid")

        def _rebuild_sync() -> int:
            doc_repo = SqlDocumentStorage()
            embedder = hooks._build_dense_embedder()
            from local_rag_backend.infrastructure.persistence.faiss.manifest import (
                purge_index_artifacts,
            )

            purge_index_artifacts(index_path=settings.index_path, id_map_path=settings.id_map_path)
            vec = FaissVectorStorage(
                index_path=settings.index_path, id_map_path=settings.id_map_path, dim=embedder.dim
            )
            return rebuild_index_from_db(doc_repo=doc_repo, vec_repo=vec, embedder=embedder)

        mutation_attempted = True
        n = hooks._run_with_multi_store_write_lock(_rebuild_sync)
        click.echo(f"[OK] Rebuilt index with {n} vectors.")
    except Exception as e:
        click.echo(f"[ERROR] Error rebuilding index: {e}", err=True)
        raise SystemExit(1)
    finally:
        if mutation_attempted:
            _hooks()._reset_rag_service_best_effort()


@click.command("status")
def status_cmd() -> None:
    """Display system status and configuration."""
    from local_rag_backend.infrastructure.persistence.sqlalchemy import base as db_base

    _hooks()._ensure_sqlite_schema_for_cli()

    title_fg = "cyan"
    key_fg = "blue"

    click.secho("🔍 Intrinsical RAG Prototype - System Status", fg=title_fg, bold=True)
    click.echo()

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
                    sample = ", ".join(
                        f"{m.get('key')}={m.get('actual')} (expected {m.get('expected')})"
                        for m in mm[:3]
                        if isinstance(m, dict)
                    )
                    click.echo(
                        f"  {click.style('Manifest drift:', fg=key_fg, bold=True)} [ERROR] {sample}"
                    )

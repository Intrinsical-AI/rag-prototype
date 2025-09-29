# src/cli.py
"""
CLI entry points for Intrinsical RAG Prototype.

This module provides command-line interfaces for common operations like
starting the server, building indices, and bootstrapping data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import uvicorn

from local_rag_backend import __version__
from local_rag_backend.settings import settings


@click.group()
@click.version_option(version=__version__, prog_name="intrinsical-rag-prototype")
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
    except Exception:
        sys.exit(1)


@cli.command()
def status() -> None:
    """Display system status and configuration."""
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
        f"  {click.style('OpenAI Model:', fg=key_fg, bold=True)} {settings.openai_model or 'Not set'}"
    )
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


if __name__ == "__main__":
    cli()

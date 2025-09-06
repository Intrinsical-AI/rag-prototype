"""
CLI entry points for Intrinsical RAG Prototype.

This module provides command-line interfaces for common operations like
starting the server, building indices, and bootstrapping data.
"""

import sys
from pathlib import Path
from typing import Optional

import click
import uvicorn

from local_rag_backend.settings import settings
from local_rag_backend import __version__


@click.group()
@click.version_option(version=__version__, prog_name="intrinsical-rag-prototype")
def cli():
    """Intrinsical RAG Prototype - Production-ready RAG system with hexagonal architecture."""
    pass


@cli.command()
@click.option("--host", default=None, help="Host IP address")
@click.option("--port", default=None, type=int, help="Port number")
@click.option("--reload/--no-reload", default=None, help="Enable auto-reload")
@click.option("--log-level", default=None, type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Log level")
def server(host: Optional[str], port: Optional[int], reload: Optional[bool], log_level: Optional[str]):
    """Start the RAG FastAPI server."""
    # Use CLI args or fall back to settings
    server_host = host or settings.app_host
    server_port = port or settings.app_port
    server_reload = reload if reload is not None else settings.debug
    server_log_level = (log_level or settings.log_level).lower()
    
    click.echo(f"🚀 Starting RAG server on {server_host}:{server_port}")
    click.echo(f"📊 Retrieval mode: {settings.retrieval_mode}")
    click.echo(f"🔧 Debug mode: {server_reload}")
    click.echo(f"📝 Log level: {server_log_level.upper()}")
    
    uvicorn.run(
        "local_rag_backend.app.main:app",
        host=server_host,
        port=server_port,
        reload=server_reload,
        log_level=server_log_level
    )


@cli.command("build-index")
def build_index():
    """Build FAISS index from existing documents."""
    try:
        # Import here to avoid circular imports
        from local_rag_backend.scripts.build_index import main as build_main
        
        click.echo("🔨 Building FAISS index...")
        with click.progressbar(length=1, label="Building index") as bar:
            build_main()
            bar.update(1)
        click.echo("✅ Index built successfully!")
    except Exception as e:
        click.echo(f"❌ Error building index: {e}", err=True)
        sys.exit(1)


@cli.command()
def bootstrap():
    """Bootstrap database with sample data."""
    try:
        # Import here to avoid circular imports
        from local_rag_backend.scripts.bootstrap import main as bootstrap_main
        
        click.echo("🌱 Bootstrapping database with sample data...")
        with click.progressbar(length=1, label="Bootstrapping") as bar:
            bootstrap_main()
            bar.update(1)
        click.echo("✅ Bootstrap completed successfully!")
    except Exception as e:
        click.echo(f"❌ Error during bootstrap: {e}", err=True)
        sys.exit(1)


@cli.command()
def status():
    """Show system status and configuration."""
    click.echo("🧠 Intrinsical RAG Prototype - System Status")
    click.echo("=" * 50)
    
    # Configuration
    click.echo(f"🌐 Host: {settings.app_host}:{settings.app_port}")
    click.echo(f"🔍 Retrieval Mode: {settings.retrieval_mode}")
    click.echo(f"🔧 Debug: {settings.debug}")
    click.echo(f"📁 Data Directory: {settings.data_dir}")
    click.echo(f"🗄️  Database: {settings.sqlite_url}")
    click.echo(f"📊 FAISS Index: {settings.index_path}")
    
    # LLM Configuration
    click.echo(f"🤖 Ollama Enabled: {settings.ollama_enabled}")
    if settings.ollama_enabled:
        click.echo(f"   └── URL: {settings.ollama_base_url}")
        click.echo(f"   └── Model: {settings.ollama_model}")
    click.echo(f"🧠 OpenAI Model: {settings.openai_model}")
    click.echo(f"🔤 Embedding Model: {settings.st_embedding_model}")
    
    # File Status
    click.echo("\n📋 File Status:")
    db_path = settings.get_database_path()
    db_status = "✅ YES" if db_path.exists() else "❌ NO"
    click.echo(f"   Database: {db_status} ({db_path})")
    
    index_path = Path(settings.index_path)
    index_status = "✅ YES" if index_path.exists() else "❌ NO"
    click.echo(f"   FAISS index: {index_status} ({index_path})")
    
    csv_path = Path(settings.faq_csv)
    csv_status = "✅ YES" if csv_path.exists() else "❌ NO"
    click.echo(f"   Sample data: {csv_status} ({csv_path})")


# Entry point functions for setuptools
def rag_server():
    """Entry point for rag-server command."""
    cli(["server", *sys.argv[1:]])


def rag_build_index():
    """Entry point for rag-build-index command."""
    cli(["build-index", *sys.argv[1:]])


def rag_bootstrap():
    """Entry point for rag-bootstrap command."""
    cli(["bootstrap", *sys.argv[1:]])


def rag_status():
    """Entry point for rag-status command."""
    cli(["status", *sys.argv[1:]])


if __name__ == "__main__":
    cli()

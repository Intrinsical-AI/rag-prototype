from __future__ import annotations

import click

from local_rag_backend.settings import get_settings


@click.command("server")
def server_cmd() -> None:
    """Start the RAG FastAPI server using settings from config.yaml."""
    import uvicorn

    click.echo(f"🚀 Starting server on {get_settings().app_host}:{get_settings().app_port}...")
    click.echo(f"   - Mode: {'Development (reload)' if get_settings().debug else 'Production'}")
    click.echo(f"   - Log level: {get_settings().log_level.upper()}")
    click.echo(f"   - Retrieval: {get_settings().retrieval_mode}")

    uvicorn.run(
        "local_rag_backend.http.main:app",
        host=get_settings().app_host,
        port=get_settings().app_port,
        reload=get_settings().debug,
        log_level=get_settings().log_level.lower(),
    )

from __future__ import annotations

import click

from local_rag_backend.settings import enforce_safe_bind_config, get_settings


@click.command("server")
@click.option("--workers", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--reload/--no-reload", default=None, help="Override config debug mode.")
def server_cmd(workers: int, reload: bool | None) -> None:
    """Start the RAG FastAPI server using settings from config.yaml."""
    import uvicorn

    settings = get_settings()
    enforce_safe_bind_config(settings)
    reload_enabled = settings.debug if reload is None else reload
    if reload_enabled and workers != 1:
        raise click.UsageError("Reload requires one worker.")
    click.echo(f"🚀 Starting server on {get_settings().app_host}:{get_settings().app_port}...")
    click.echo(f"   - Mode: {'Development (reload)' if reload_enabled else 'Production'}")
    click.echo(f"   - Log level: {get_settings().log_level.upper()}")
    click.echo(f"   - Retrieval: {get_settings().retrieval_mode}")

    uvicorn.run(
        "local_rag_backend.http.main:app",
        host=get_settings().app_host,
        port=get_settings().app_port,
        reload=reload_enabled,
        workers=workers,
        log_level=get_settings().log_level.lower(),
    )

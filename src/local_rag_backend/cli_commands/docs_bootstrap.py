from __future__ import annotations

import click

from local_rag_backend.cli_commands.runtime import run_cli_mutation


@click.command("bootstrap")
def bootstrap_cmd() -> None:
    """Bootstrap database with sample data."""
    try:
        from local_rag_backend.scripts.bootstrap import main as bootstrap_main

        click.echo("[INFO] Bootstrapping database with sample data...")
        with click.progressbar(length=1, label="Bootstrapping") as bar:
            run_cli_mutation(bootstrap_main, use_lock=False, ensure_schema=False)
            bar.update(1)
        click.echo("[OK] Bootstrap completed successfully!")
    except Exception as e:
        click.echo(f"[ERROR] Error bootstrapping: {e}", err=True)
        raise SystemExit(1)

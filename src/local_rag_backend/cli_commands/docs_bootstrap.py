from __future__ import annotations

import click

from local_rag_backend.cli_commands.docs_common import _hooks, _reset_if_mutated


@click.command("bootstrap")
def bootstrap_cmd() -> None:
    """Bootstrap database with sample data."""
    mutation_attempted = False
    try:
        hooks = _hooks()
        hooks._ensure_sqlite_schema_for_cli()
        from local_rag_backend.scripts.bootstrap import main as bootstrap_main

        click.echo("[INFO] Bootstrapping database with sample data...")
        with click.progressbar(length=1, label="Bootstrapping") as bar:
            mutation_attempted = True
            hooks._run_with_multi_store_write_lock(bootstrap_main)
            bar.update(1)
        click.echo("[OK] Bootstrap completed successfully!")
    except Exception as e:
        click.echo(f"[ERROR] Error bootstrapping: {e}", err=True)
        raise SystemExit(1)
    finally:
        _reset_if_mutated(mutation_attempted=mutation_attempted)


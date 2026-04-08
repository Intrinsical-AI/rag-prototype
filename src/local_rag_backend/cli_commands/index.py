from __future__ import annotations

from pathlib import Path

import click

from local_rag_backend.cli_commands.runtime import (
    build_dense_embedder,
    ensure_sqlite_schema_for_cli,
    get_cli_container,
    get_cli_runtime_snapshot,
    run_cli_mutation,
)


@click.command("rebuild-index")
def rebuild_index_cmd() -> None:
    """Rebuild retrieval index from the current document store (idempotent)."""
    try:
        from local_rag_backend.core.use_cases import index as index_service

        container = get_cli_container()
        runtime = get_cli_runtime_snapshot()
        if runtime.retrieval_mode not in ("dense", "dual", "hybrid"):
            raise RuntimeError("rebuild-index requires retrieval_mode=dense|dual|hybrid")
        ports = container.index_mutation_ports(
            build_embedder=build_dense_embedder,
        )

        def _rebuild_sync() -> int:
            return index_service.rebuild_index_sync(
                settings_obj=container.settings_obj,
                ports=ports,
            )

        n = run_cli_mutation(_rebuild_sync)
        click.echo(f"[OK] Rebuilt index with {n} vectors.")
    except Exception as e:
        click.echo(f"[ERROR] Error rebuilding index: {e}", err=True)
        raise SystemExit(1)


@click.command("status")
def status_cmd() -> None:
    """Display system status and configuration."""
    container = get_cli_container()
    runtime = get_cli_runtime_snapshot()
    ensure_sqlite_schema_for_cli()
    readiness_bundle = container.build_health_readiness_bundle()
    diagnostics = readiness_bundle.diagnostics

    title_fg = "cyan"
    key_fg = "blue"

    click.secho("🔍 Intrinsical RAG Prototype - System Status", fg=title_fg, bold=True)
    click.echo()

    click.secho("📋 Core Configuration", fg=title_fg, bold=True)
    click.echo(
        f"  {click.style('Host:', fg=key_fg, bold=True)} {runtime.host}:{runtime.port}"
    )
    click.echo(
        f"  {click.style('Retrieval Mode:', fg=key_fg, bold=True)} {runtime.retrieval_mode}"
    )
    click.echo(
        f"  {click.style('Persistence Backend:', fg=key_fg, bold=True)} {runtime.persistence_backend}"
    )
    click.echo(
        f"  {click.style('Debug Mode:', fg=key_fg, bold=True)} {'✅' if runtime.debug else '❌'}"
    )
    click.echo(
        f"  {click.style('Vector Backend:', fg=key_fg, bold=True)} {runtime.vector_backend}"
    )
    click.echo(
        f"  {click.style('API Key Configured:', fg=key_fg, bold=True)} {'✅' if runtime.api_key_configured else '❌'}"
    )
    click.echo(
        f"  {click.style('Public Bind Requires API Key:', fg=key_fg, bold=True)} {'✅' if runtime.public_bind_requires_api_key else '❌'}"
    )
    click.echo()

    click.secho("🤖 LLM Configuration", fg=title_fg, bold=True)
    click.echo(
        f"  {click.style('Ollama Enabled:', fg=key_fg, bold=True)} {'✅' if runtime.ollama_enabled else '❌'}"
    )
    if runtime.ollama_enabled:
        click.echo(
            f"    {click.style('URL:', fg=key_fg, bold=True)} {container.settings_obj.ollama_base_url}"
        )
        click.echo(
            f"    {click.style('Model:', fg=key_fg, bold=True)} {container.settings_obj.ollama_model}"
        )
    click.echo(
        f"  {click.style('OpenAI Enabled:', fg=key_fg, bold=True)} {'✅' if runtime.openai_enabled else '❌'}"
    )
    if runtime.openai_enabled:
        click.echo(
            f"    {click.style('Model:', fg=key_fg, bold=True)} {container.settings_obj.openai_model}"
        )
    click.echo()

    click.secho("📁 Storage Status", fg=title_fg, bold=True)
    if runtime.persistence_backend == "elasticsearch":
        click.echo(
            f"  {click.style('Elasticsearch:', fg=key_fg, bold=True)} {container.settings_obj.es_base_url or '[MISSING]'}"
        )
        for name, value in [
            ("Docs index", container.settings_obj.es_docs_index),
            ("History index", container.settings_obj.es_history_index),
            ("System index", container.settings_obj.es_system_index),
            ("Tombstones index", container.settings_obj.es_tombstones_index),
        ]:
            click.echo(f"  {click.style(name + ':', fg=key_fg, bold=True)} {value}")
    else:
        for name, path_str in [
            ("Database", container.settings_obj.sqlite_url.replace("sqlite:///", "")),
            ("FAISS index", runtime.index_path),
            ("Sample data", container.settings_obj.faq_csv),
        ]:
            path = Path(path_str)
            status_icon = "✅ YES" if path.exists() else "❌ NO"
            click.echo(f"  {click.style(name + ':', fg=key_fg, bold=True)} {status_icon}")
            click.echo(f"    Path: {path}")

    click.echo()
    click.secho("📊 Data & Index Diagnostics", fg=title_fg, bold=True)

    docs_count: int | None = None
    try:
        docs_count = diagnostics.get_documents_count()
        click.echo(f"  {click.style('Documents:', fg=key_fg, bold=True)} {docs_count}")
    except Exception as e:
        click.echo(f"  {click.style('Documents:', fg=key_fg, bold=True)} [ERROR] {e!s}")

    try:
        hist = diagnostics.get_history_count()
        click.echo(f"  {click.style('History:', fg=key_fg, bold=True)} {hist}")
    except Exception as e:
        click.echo(f"  {click.style('History:', fg=key_fg, bold=True)} [WARN] {e!s}")

    if runtime.retrieval_mode in ("dense", "dual", "hybrid"):
        try:
            stats = diagnostics.get_retrieval_index_stats(
                index_path=runtime.index_path,
                id_map_path=runtime.id_map_path,
                vector_backend=runtime.vector_backend,
                dim=None,
                expected_manifest=readiness_bundle.expected_manifest,
            )
        except Exception as e:
            click.echo(f"  {click.style('Index:', fg=key_fg, bold=True)} [ERROR] {e!s}")
            return
        status_txt = str(stats.get("status"))
        if status_txt == "ok":
            click.echo(
                f"  {click.style('Index:', fg=key_fg, bold=True)} "
                f"{stats.get('vectors')} vectors "
                f"(dim={stats.get('dim')}, backend={stats.get('backend')})"
            )
            if runtime.persistence_backend != "elasticsearch":
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
            if status_txt == "drift" and runtime.persistence_backend != "elasticsearch":
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

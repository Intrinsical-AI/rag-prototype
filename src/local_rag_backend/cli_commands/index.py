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


def _echo_status_value(*, label: str, value: object, key_fg: str) -> None:
    click.echo(f"  {click.style(label + ':', fg=key_fg, bold=True)} {value}")


def _echo_nested_status_value(*, label: str, value: object, key_fg: str) -> None:
    click.echo(f"    {click.style(label + ':', fg=key_fg, bold=True)} {value}")


def _echo_path_status(*, label: str, path_str: str, key_fg: str) -> None:
    path = Path(path_str)
    status_icon = "✅ YES" if path.exists() else "❌ NO"
    _echo_status_value(label=label, value=status_icon, key_fg=key_fg)
    click.echo(f"    Path: {path}")


def _require_rebuild_runtime(retrieval_mode: str) -> None:
    if retrieval_mode not in ("dense", "dual", "hybrid"):
        raise RuntimeError("rebuild-index requires retrieval_mode=dense|dual|hybrid")


@click.command("rebuild-index")
def rebuild_index_cmd() -> None:
    """Rebuild retrieval index from the current document store (idempotent)."""
    try:
        from local_rag_backend.core.use_cases import index as index_service

        container = get_cli_container()
        runtime = get_cli_runtime_snapshot()
        _require_rebuild_runtime(runtime.retrieval_mode)
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
    _echo_status_value(label="Host", value=f"{runtime.host}:{runtime.port}", key_fg=key_fg)
    _echo_status_value(label="Retrieval Mode", value=runtime.retrieval_mode, key_fg=key_fg)
    _echo_status_value(
        label="Persistence Backend",
        value=runtime.persistence_backend,
        key_fg=key_fg,
    )
    _echo_status_value(
        label="Debug Mode",
        value=("✅" if runtime.debug else "❌"),
        key_fg=key_fg,
    )
    _echo_status_value(label="Vector Backend", value=runtime.vector_backend, key_fg=key_fg)
    _echo_status_value(
        label="API Key Configured",
        value=("✅" if runtime.api_key_configured else "❌"),
        key_fg=key_fg,
    )
    _echo_status_value(
        label="Public Bind Requires API Key",
        value=("✅" if runtime.public_bind_requires_api_key else "❌"),
        key_fg=key_fg,
    )
    click.echo()

    click.secho("🤖 LLM Configuration", fg=title_fg, bold=True)
    _echo_status_value(
        label="Ollama Enabled",
        value=("✅" if runtime.ollama_enabled else "❌"),
        key_fg=key_fg,
    )
    if runtime.ollama_enabled:
        _echo_nested_status_value(
            label="URL",
            value=container.settings_obj.ollama_base_url,
            key_fg=key_fg,
        )
        _echo_nested_status_value(
            label="Model",
            value=container.settings_obj.ollama_model,
            key_fg=key_fg,
        )
    _echo_status_value(
        label="OpenAI Enabled",
        value=("✅" if runtime.openai_enabled else "❌"),
        key_fg=key_fg,
    )
    if runtime.openai_enabled:
        _echo_nested_status_value(
            label="Model",
            value=container.settings_obj.openai_model,
            key_fg=key_fg,
        )
    click.echo()

    click.secho("📁 Storage Status", fg=title_fg, bold=True)
    if runtime.persistence_backend == "elasticsearch":
        _echo_status_value(
            label="Elasticsearch",
            value=(container.settings_obj.es_base_url or "[MISSING]"),
            key_fg=key_fg,
        )
        for name, value in [
            ("Docs index", container.settings_obj.es_docs_index),
            ("History index", container.settings_obj.es_history_index),
            ("System index", container.settings_obj.es_system_index),
            ("Tombstones index", container.settings_obj.es_tombstones_index),
        ]:
            _echo_status_value(label=name, value=value, key_fg=key_fg)
    else:
        for name, path_str in [
            ("Database", container.settings_obj.sqlite_url.replace("sqlite:///", "")),
            ("FAISS index", runtime.index_path),
            ("Sample data", container.settings_obj.faq_csv),
        ]:
            _echo_path_status(label=name, path_str=path_str, key_fg=key_fg)

    click.echo()
    click.secho("📊 Data & Index Diagnostics", fg=title_fg, bold=True)

    docs_count: int | None = None
    try:
        docs_count = diagnostics.get_documents_count()
        _echo_status_value(label="Documents", value=docs_count, key_fg=key_fg)
    except Exception as e:
        _echo_status_value(label="Documents", value=f"[ERROR] {e!s}", key_fg=key_fg)

    try:
        hist = diagnostics.get_history_count()
        _echo_status_value(label="History", value=hist, key_fg=key_fg)
    except Exception as e:
        _echo_status_value(label="History", value=f"[WARN] {e!s}", key_fg=key_fg)

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

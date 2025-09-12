"""
CLI entry points for Intrinsical RAG Prototype.

This module provides command-line interfaces for common operations like
starting the server, building indices, and bootstrapping data.
"""

import logging
import sys
from pathlib import Path

import click
import uvicorn

from local_rag_backend import __version__
from local_rag_backend.settings import settings

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__, prog_name="intrinsical-rag-prototype")
def cli() -> None:
    """Intrinsical RAG Prototype - Production-ready RAG system with hexagonal architecture."""
    pass


@cli.command()
@click.option("--host", default=None, help="Host IP address")
@click.option("--port", default=None, type=int, help="Port number")
@click.option("--reload/--no-reload", default=None, help="Enable auto-reload")
@click.option(
    "--log-level",
    default=None,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Log level",
)
def server(host: str | None, port: int | None, reload: bool | None, log_level: str | None) -> None:
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
        log_level=server_log_level,
    )


@cli.command("build-index")
def build_index() -> None:
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
def bootstrap() -> None:
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


@cli.command("ingest-files")
@click.option("--root", required=True, type=click.Path(exists=True, file_okay=False))
@click.option(
    "--pattern",
    multiple=True,
    default=["**/*.md", "**/*.txt"],
    help="Glob patterns to include (repeatable)",
)
@click.option(
    "--dedup/--no-dedup",
    default=True,
    show_default=True,
    help="Deduplicate identical texts within the run",
)
@click.option(
    "--incremental/--no-incremental",
    default=True,
    show_default=True,
    help="Skip files unchanged since last run using a state file",
)
@click.option(
    "--incremental-strategy",
    type=click.Choice(["mtime", "hash"]),
    default="mtime",
    show_default=True,
    help="Strategy to detect changes: modified time (fast) or content hash (robust)",
)
@click.option(
    "--state-path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to incremental state JSON (defaults to data/.ingest_state.json)",
)
def ingest_files(
    root: str,
    pattern: tuple[str, ...],
    dedup: bool,
    incremental: bool,
    incremental_strategy: str,
    state_path: str | None,
) -> None:
    """Ingesta e indexación desde un directorio."""
    import json
    from pathlib import Path as _Path

    from local_rag_backend.core.services.etl import ETLService
    from local_rag_backend.core.services.ingestion import (
        IngestionPipeline,
        default_chunker,
        default_formatter,
        default_preprocess,
    )
    from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
        SentenceTransformerEmbedder,
    )
    from local_rag_backend.infrastructure.ingestion.loaders import (
        DirectoryLoader,
        UniqueLoader,
    )
    from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
    from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

    # Build candidate file list (with optional incremental filtering)
    root_path = _Path(root)
    all_files: list[_Path] = []
    seen: set[_Path] = set()
    for pat in pattern:
        for p in root_path.rglob(pat):
            if p.is_file() and p not in seen:
                seen.add(p)
                all_files.append(p)

    state_default = str(_Path(settings.data_dir) / ".ingest_state.json")
    state_file = _Path(state_path or state_default)
    old_state_raw: dict[str, dict[str, float | str]] | dict[str, float] = {}
    if incremental and state_file.exists():
        try:
            old_state_raw = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            old_state_raw = {}

    def _get_old_mtime(key: str) -> float | None:
        val = old_state_raw.get(key)
        if isinstance(val, dict):
            mtime = val.get("mtime")
            return float(mtime) if mtime is not None else None
        if isinstance(val, int | float):
            return float(val)
        return None

    def _get_old_hash(key: str) -> str | None:
        val = old_state_raw.get(key)
        if isinstance(val, dict):
            h = val.get("hash")
            return str(h) if h is not None else None
        return None

    computed_hash: dict[str, str] = {}

    def _sha256(path: _Path) -> str:
        import hashlib as _hashlib

        h = _hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def file_changed(p: _Path) -> bool:
        if not incremental:
            return True
        key = str(p.resolve())
        if incremental_strategy == "mtime":
            mtime = p.stat().st_mtime
            last_mtime = _get_old_mtime(key)
            return last_mtime is None or mtime > float(last_mtime)
        else:  # hash strategy
            try:
                h = _sha256(p)
                computed_hash[key] = h
            except Exception:
                # if hashing fails, fallback to mtime
                h = None
            old_h = _get_old_hash(key)
            return h is None or old_h != h

    files_to_process = [p for p in all_files if file_changed(p)]
    if not files_to_process:
        click.echo("i  No files changed since last run (incremental enabled). Nothing to ingest.")
        return

    loader: DirectoryLoader | UniqueLoader = DirectoryLoader(root, files=files_to_process)
    if dedup:
        loader = UniqueLoader(loader)
    doc_repo = SqlDocumentStorage()

    if settings.retrieval_mode in ["dense", "hybrid"]:
        embedder = SentenceTransformerEmbedder(model_name=settings.st_embedding_model)
        vec = FaissVectorStorage(settings.index_path, settings.id_map_path, dim=embedder.dim)
        etl = ETLService(doc_repo, vec, embedder)
        pipeline = IngestionPipeline(
            loader,
            etl,
            chunk=default_chunker(settings.ingest_chunk_chars, settings.ingest_chunk_overlap),
        )
        processed_ids = pipeline.run()
    else:
        # Sparse: solo SQL (igual que en bootstrap.py)
        buf: list[str] = []
        processed_ids = []
        batch = 128
        chunk = default_chunker(settings.ingest_chunk_chars, settings.ingest_chunk_overlap)
        for item in loader.load():
            clean = default_preprocess(item.text, dict(item.metadata) if item.metadata else None)
            for c in chunk(clean, dict(item.metadata) if item.metadata else None):
                buf.append(default_formatter(c, dict(item.metadata) if item.metadata else None))
                if len(buf) >= batch:
                    processed_ids.extend(doc_repo.store_documents(buf))
                    buf.clear()
        if buf:
            processed_ids.extend(doc_repo.store_documents(buf))
    click.echo(f"✅ Ingested {len(processed_ids)} chunks from {len(files_to_process)} files")

    # Update incremental state
    if incremental:
        # normalize previous state
        new_state: dict[str, dict] = {}  # type: ignore
        for k, v in old_state_raw.items():
            if isinstance(v, dict):
                new_state[k] = {"mtime": float(v.get("mtime", 0.0)), "hash": str(v.get("hash", ""))}
            elif isinstance(v, int | float):
                new_state[k] = {"mtime": float(v), "hash": new_state.get(k, {}).get("hash", "")}

        # update only processed files
        for p in files_to_process:
            key = str(p.resolve())
            try:
                mtime = Path(p).stat().st_mtime
                hash = computed_hash.get(key)
                if hash is None and incremental_strategy == "hash":
                    # compute if missing
                    hash = _sha256(Path(p))
                new_state[key] = {
                    "mtime": float(mtime),
                    "hash": hash or new_state.get(key, {}).get("hash", ""),
                }
            except Exception as e:
                logger.warning("Failed to process file %s: %s", p, str(e))
                continue
        try:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text(
                json.dumps(new_state, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            click.echo(f"📝 Updated state file: {state_file}")
        except Exception as e:
            click.echo(f"⚠️  Could not update state file: {e}", err=True)


@cli.command("ingest-web")
@click.option("--url", "urls", multiple=True, help="URL a ingerir (repetir para varias)")
@click.option(
    "--from-file",
    type=click.Path(exists=True, dir_okay=False),
    help="Fichero con URLs (una por línea)",
)
@click.option("--timeout", type=int, default=20, show_default=True, help="Timeout por petición")
@click.option(
    "--workers", type=int, default=1, show_default=True, help="Concurrencia durante la descarga"
)
@click.option(
    "--dedup/--no-dedup",
    default=True,
    show_default=True,
    help="Deduplicar textos idénticos en esta ejecución",
)
def ingest_web(
    urls: tuple[str, ...], from_file: str | None, timeout: int, workers: int, dedup: bool
) -> None:
    """Ingesta e indexación desde URLs (web scraping ligero con trafilatura)."""
    from local_rag_backend.core.services.etl import ETLService
    from local_rag_backend.core.services.ingestion import (
        IngestionPipeline,
        default_chunker,
        default_formatter,
        default_preprocess,
    )
    from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
        SentenceTransformerEmbedder,
    )
    from local_rag_backend.infrastructure.ingestion.loaders import (
        UniqueLoader,
        WebPageLoader,
    )
    from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
    from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

    # Reunir URLs de CLI y/o fichero
    url_list: list[str] = list(urls)
    if from_file:
        try:
            with open(from_file, encoding="utf-8") as fh:
                for line in fh:
                    u = line.strip()
                    if u:
                        url_list.append(u)
        except Exception as e:
            click.echo(f"❌ No se pudo leer el fichero de URLs: {e}", err=True)
            sys.exit(1)

    # Validación básica
    if not url_list:
        click.echo("❌ Debes proporcionar al menos una URL con --url o --from-file", err=True)
        sys.exit(2)

    loader: WebPageLoader | UniqueLoader = WebPageLoader(url_list, timeout=timeout, workers=workers)
    if dedup:
        loader = UniqueLoader(loader)
    doc_repo = SqlDocumentStorage()

    if settings.retrieval_mode in ["dense", "hybrid"]:
        embedder = SentenceTransformerEmbedder(model_name=settings.st_embedding_model)
        vec = FaissVectorStorage(settings.index_path, settings.id_map_path, dim=embedder.dim)
        etl = ETLService(doc_repo, vec, embedder)
        pipeline = IngestionPipeline(
            loader,
            etl,
            chunk=default_chunker(settings.ingest_chunk_chars, settings.ingest_chunk_overlap),
        )
        processed_ids = pipeline.run()
    else:
        # Sparse: solo SQL
        buf: list[str] = []
        processed_ids = []
        batch = 128
        chunk = default_chunker(settings.ingest_chunk_chars, settings.ingest_chunk_overlap)
        for item in loader.load():
            clean = default_preprocess(item.text, dict(item.metadata) if item.metadata else None)
            for c in chunk(clean, dict(item.metadata) if item.metadata else None):
                buf.append(default_formatter(c, dict(item.metadata) if item.metadata else None))
                if len(buf) >= batch:
                    processed_ids.extend(doc_repo.store_documents(buf))
                    buf.clear()
        if buf:
            processed_ids.extend(doc_repo.store_documents(buf))
    click.echo(f"✅ Ingested {len(processed_ids)} chunks from {len(url_list)} URLs")


@cli.command()
def status() -> None:
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


@cli.command("health")
def health_cmd() -> None:
    from local_rag_backend.app.factory import get_rag_service

    try:
        get_rag_service()
        click.echo("ok")
    except Exception as e:
        click.echo(f"error: {e}", err=True)
        sys.exit(1)


# Entry point functions for setuptools
def _run_cli(args: list[str]) -> None:
    """Helper to run CLI with proper argument parsing."""
    cli.main(args=args, standalone_mode=False)


def rag_server() -> None:
    """Entry point for rag-server command."""
    _run_cli(["server", *sys.argv[1:]])


def rag_build_index() -> None:
    """Entry point for rag-build-index command."""
    _run_cli(["build-index", *sys.argv[1:]])


def rag_bootstrap() -> None:
    """Entry point for rag-bootstrap command."""
    _run_cli(["bootstrap", *sys.argv[1:]])


def rag_ingest_files() -> None:
    """Entry point for rag-ingest-files command."""
    _run_cli(["ingest-files", *sys.argv[1:]])


def rag_ingest_web() -> None:
    """Entry point for rag-ingest-web command."""
    _run_cli(["ingest-web", *sys.argv[1:]])


def rag_health() -> None:
    """Entry point for rag-health command."""
    _run_cli(["health", *sys.argv[1:]])


def rag_status() -> None:
    """Entry point for rag-status command."""
    _run_cli(["status", *sys.argv[1:]])


if __name__ == "__main__":
    cli()

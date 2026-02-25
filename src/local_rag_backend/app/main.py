"""
FastAPI application entry point.
"""

from __future__ import annotations

import logging
import mimetypes
import sys
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, Response

from local_rag_backend.app.api_router import router
from local_rag_backend.app.dependencies import get_rag_service
from local_rag_backend.app.errors import NotFoundError
from local_rag_backend.app.factory import reset_app_context
from local_rag_backend.app.http.exception_handlers import register_exception_handlers
from local_rag_backend.app.middleware import MetricsMiddleware, get_metrics
from local_rag_backend.app.security import enforce_safe_bind_config, require_api_key
from local_rag_backend.infrastructure.persistence.sql import base as db_base
from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from importlib.resources.abc import Traversable

_LOG_LEVEL = getattr(logging, settings.log_level, logging.INFO)
logging.basicConfig(level=_LOG_LEVEL, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Frontend directory in the source tree (works for editable installs / running from repo)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown events."""
    logger.info("Initializing RAG service...")
    enforce_safe_bind_config()
    # Ensure the data directory exists (SQLite cannot create parent directories).
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    # Use the module reference so tests can monkeypatch `db_base.engine` / `db_base.SessionLocal`.
    db_base.ensure_sqlite_schema_compatible(engine_to_use=db_base.engine)
    # Best-effort preload: don't prevent the API from starting just because an LLM
    # provider isn't configured yet (readiness endpoint should report not_ready).
    try:
        await get_rag_service()
    except Exception as e:
        logger.warning("RAG service preload failed (will initialize lazily): %s", e)
        # Avoid caching a half-initialized runtime context after preload failures.
        reset_app_context()
    logger.info("Service initialized.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Local RAG Demo", lifespan=lifespan)
register_exception_handlers(app)

# CORS: keep permissive defaults ONLY in debug mode.
cors_allow_origins = ["*"] if settings.debug else list(settings.cors_allow_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    # `*` + credentials is invalid per the CORS spec; browsers will ignore it.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.enable_monitoring:
    app.add_middleware(MetricsMiddleware)

app.include_router(router, prefix="/api", dependencies=[Depends(require_api_key)])


@app.get(
    "/metrics",
    response_class=PlainTextResponse,
    include_in_schema=False,
    dependencies=[Depends(require_api_key)],
)
async def metrics_endpoint() -> PlainTextResponse:
    """Prometheus metrics endpoint."""
    content, content_type = get_metrics()
    return PlainTextResponse(content=content, media_type=content_type)


def get_frontend_path() -> Traversable | Path | None:
    """Try to find the index.html file, first in package resources, then in repo structure."""
    try:
        # 1. Packaged resource (for installed distributions)
        pkg_path = resources.files("local_rag_backend.frontend").joinpath("index.html")
        if pkg_path.is_file():
            return pkg_path
    except (ImportError, AttributeError):
        pass  # Fallback to repo structure

    # 2. Repo structure (for development)
    repo_path = FRONTEND_DIR / "index.html"
    if repo_path.is_file():
        return repo_path
    return None


def _get_frontend_asset(asset_path: str) -> tuple[bytes, str]:
    """
    Load a frontend asset (css/js/etc.), first from packaged resources, then from repo structure.
    """
    posix = PurePosixPath(asset_path)
    if posix.is_absolute() or ".." in posix.parts:
        raise NotFoundError("Asset not found.")

    # 1) Packaged assets
    try:
        pkg_root = resources.files("local_rag_backend.frontend")
        pkg_file = pkg_root.joinpath(*posix.parts)
        if pkg_file.is_file():
            data = pkg_file.read_bytes()
            mt = mimetypes.guess_type(posix.name)[0] or "application/octet-stream"
            return data, mt
    except (ImportError, AttributeError):
        pass

    # 2) Repo assets
    fs_file = FRONTEND_DIR.joinpath(*posix.parts)
    if fs_file.is_file():
        data = fs_file.read_bytes()
        mt = mimetypes.guess_type(fs_file.name)[0] or "application/octet-stream"
        return data, mt

    raise NotFoundError("Asset not found.")


@app.get("/assets/{asset_path:path}", include_in_schema=False)
async def serve_frontend_assets(asset_path: str) -> Response:
    """Serve packaged frontend assets (css/js) for the SPA."""
    data, media_type = _get_frontend_asset(asset_path)
    return Response(content=data, media_type=media_type)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend() -> HTMLResponse:
    """Serve the single-page frontend application."""
    index_path = get_frontend_path()
    if not index_path:
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)

    try:
        html_content = index_path.read_text(encoding="utf-8")
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error reading frontend file: {e}")
        return HTMLResponse("<h1>Failed to load frontend</h1>", status_code=500)


if __name__ == "__main__":
    uvicorn.run(
        "local_rag_backend.app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )

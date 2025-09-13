# src/app/main.py
import logging
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse

from local_rag_backend.app.api_router import router
from local_rag_backend.app.dependencies import get_rag_service
from local_rag_backend.app.middleware import get_metrics
from local_rag_backend.infrastructure.persistence.sqlalchemy.base import Base as AppDeclarativeBase
from local_rag_backend.infrastructure.persistence.sqlalchemy.base import engine as global_app_engine
from local_rag_backend.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)  # logger after de basicConfig


# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(_app_instance: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Lifespan startup: Checking/Creating database tables...")
    AppDeclarativeBase.metadata.create_all(bind=global_app_engine)
    logger.info("Lifespan startup: Database tables checked/created.")

    logger.info("Lifespan startup: Initializing RAG service...")
    get_rag_service()
    logger.info("Lifespan startup: RAG service initialized.")
    yield
    logger.info("Lifespan shutdown: Cleaning up resources (if any)...")


app = FastAPI(title="Local RAG Demo", lifespan=lifespan)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def metrics_endpoint() -> PlainTextResponse:
    """Prometheus metrics endpoint."""
    content, content_type = get_metrics()
    return PlainTextResponse(content=content, media_type=content_type)


CURRENT_FILE_PATH = Path(__file__).resolve()
SRC_APP_DIR = CURRENT_FILE_PATH.parent
SRC_DIR = SRC_APP_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
FRONTEND_DIR = PROJECT_ROOT_DIR / "frontend"


@app.get("/", response_class=HTMLResponse)
async def read_root(_request: Request) -> HTMLResponse:
    # 1) Try to serve packaged frontend (installed package)
    try:
        pkg_index = resources.files("local_rag_backend.frontend").joinpath("index.html")
        if pkg_index.is_file():
            with pkg_index.open("r", encoding="utf-8") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        # Log error and fallback to repository frontend
        logger.warning(f"Failed to load packaged frontend: {e}")

    # 2) Fallback: serve from repository root (developer mode)
    index_html_path = FRONTEND_DIR / "index.html"
    if not index_html_path.is_file():
        logger.error(f"Frontend file not found at {index_html_path}")
        return HTMLResponse(
            content="<h1>Frontend not found</h1><p>Please check server configuration.</p>",
            status_code=404,
        )

    try:
        with open(index_html_path, encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logger.error(f"Could not read frontend file {index_html_path}: {e}", exc_info=True)
        return HTMLResponse(content="<h1>Error serving frontend</h1>", status_code=500)


if __name__ == "__main__":
    uvicorn.run(
        "local_rag_backend.app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )

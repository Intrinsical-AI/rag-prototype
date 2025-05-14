# src/app/main.py
import logging
import sys 
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from pathlib import Path
from src.app.api_router import router
from src.app.dependencies import init_rag_service
from src.db.base import Base as AppDeclarativeBase
from src.db.base import engine as global_app_engine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__) # logger after de basicConfig

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logger.info("Lifespan startup: Checking/Creating database tables...")
    AppDeclarativeBase.metadata.create_all(bind=global_app_engine)
    logger.info("Lifespan startup: Database tables checked/created.")

    logger.info("Lifespan startup: Initializing RAG service...")
    init_rag_service()
    logger.info("Lifespan startup: RAG service initialized.")
    yield
    logger.info("Lifespan shutdown: Cleaning up resources (if any)...")

app = FastAPI(
    title="Local RAG Demo",
    lifespan=lifespan
)


app.include_router(router, prefix="/api")

CURRENT_FILE_PATH = Path(__file__).resolve() 
SRC_APP_DIR = CURRENT_FILE_PATH.parent       
SRC_DIR = SRC_APP_DIR.parent                 
PROJECT_ROOT_DIR = SRC_DIR.parent            
FRONTEND_DIR = PROJECT_ROOT_DIR / "frontend" 


@app.get("/", response_class=HTMLResponse)
async def serve_frontend_route(request: Request): 
    index_html_path = FRONTEND_DIR / "index.html"
    if not index_html_path.is_file(): 
        # Cambiar print a logger.error
        logger.error(f"Frontend file not found at {index_html_path}")
        return HTMLResponse(content="<h1>Frontend not found</h1><p>Please check server configuration.</p>", status_code=404)
    
    try:
        with open(index_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e: 
        # Cambiar print a logger.error con exc_info
        logger.error(f"Could not read frontend file {index_html_path}: {e}", exc_info=True)
        return HTMLResponse(content="<h1>Error serving frontend</h1>", status_code=500)

if __name__ == "__main__":
    uvicorn.run("src.app.main:app", host="0.0.0.0", port=8000, reload=True)
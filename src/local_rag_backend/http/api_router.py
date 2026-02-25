# src/local_rag_backend/app/api_router.py
"""
Root API router composition.
"""

from __future__ import annotations

from fastapi import APIRouter

from local_rag_backend.http.routers.docs import router as docs_router
from local_rag_backend.http.routers.health import router as health_router
from local_rag_backend.http.routers.index import router as index_router
from local_rag_backend.http.routers.meta import router as meta_router
from local_rag_backend.http.routers.openrouter import router as openrouter_router
from local_rag_backend.http.routers.rag_router import router as rag_router

router = APIRouter()
router.include_router(health_router)
router.include_router(rag_router)
router.include_router(docs_router)
router.include_router(index_router)
router.include_router(openrouter_router)
router.include_router(meta_router)

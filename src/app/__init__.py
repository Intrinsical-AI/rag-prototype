"""
File: src/app/__init__.py
Path: src/app/__init__.py
FastAPI application module.
"""

from .main import app
from .dependencies import get_rag_service

__all__ = [
    "app",
    "get_rag_service"
] 
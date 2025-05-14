"""
File: src/db/__init__.py
Path: src/db/__init__.py
Database module for the local RAG backend.
"""

from .base import Base, get_db, SessionLocal, engine
from .models import Document
from .crud import get_documents, add_documents

__all__ = [
    "Base", 
    "get_db",
    "SessionLocal",
    "engine",
    "Document",
    "get_documents",
    "add_documents"
] 
"""
File: src/adapters/__init__.py
Path: src/adapters/__init__.py
Adapters module for external service integrations.
"""

from .embeddings import OpenAIEmbedder, SentenceTransformerEmbedder

__all__ = ["OpenAIEmbedder", "SentenceTransformerEmbedder"]

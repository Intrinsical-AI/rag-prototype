"""
File: src/adapters/embeddings/__init__.py
Path: src/adapters/embeddings/__init__.py
Embedding adapters for text embedding generation.
"""

from .openai import OpenAIEmbedder
from .sentence_transformers import SentenceTransformerEmbedder

__all__ = ["OpenAIEmbedder", "SentenceTransformerEmbedder"]

"""
File: src/adapters/retrieval/__init__.py
Path: src/adapters/retrieval/__init__.py
Retrieval adapters for document retrieval.
"""

from .dense_faiss import DenseFaissRetriever
from .sparse_bm25 import SparseBM25Retriever
from .hybrid import HybridRetriever

__all__ = ["DenseFaissRetriever", "SparseBM25Retriever", "HybridRetriever"]

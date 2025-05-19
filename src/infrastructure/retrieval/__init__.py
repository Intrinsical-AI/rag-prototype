"""
File: src/adapters/retrieval/__init__.py
Path: src/adapters/retrieval/__init__.py
Retrieval adapters for document retrieval.
"""

from .dense_faiss import DenseFaissRetriever
from .hybrid import HybridRetriever
from .sparse_bm25 import SparseBM25Retriever

__all__ = ["DenseFaissRetriever", "SparseBM25Retriever", "HybridRetriever"]

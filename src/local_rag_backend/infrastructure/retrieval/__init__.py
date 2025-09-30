"""
RAG Prototype - Intrinsical-AI (c) 2025
Author: Pablo Pintor
License: MIT

Retrieval adapters for document retrieval.
"""

from .dense_faiss import DenseFaissRetriever
from .hybrid import HybridRetriever
from .sparse_bm25 import SparseBM25Retriever

__all__ = ["DenseFaissRetriever", "HybridRetriever", "SparseBM25Retriever"]

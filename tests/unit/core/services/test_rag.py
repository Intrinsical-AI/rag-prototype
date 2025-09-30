"""
RAG Prototype - Intrinsical-AI (c) 2025
Author: Pablo Pintor
License: MIT

Module: RAG Service Unit Tests
Purpose: Tests for the core RAG service orchestration logic.
         Validates the complete RAG pipeline using test doubles.
"""

from __future__ import annotations

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.services.rag import RagService

# --- Test Doubles ---

class DummyRetriever:
    """Mock retriever for testing RAG service behavior."""

    def __init__(self, docs, scores):
        """Initialize with predefined documents and scores.

        Args:
            docs: List of documents to return from retrieve calls
            scores: List of scores corresponding to the documents
        """
        self._docs, self._scores = docs, scores
        self.last_query = None

    def retrieve(self, query, k=3):
        """Mock retrieve method that returns predefined results.

        Args:
            query: Search query (stored for verification)
            k: Maximum results to return

        Returns:
            Tuple of (documents, scores) limited by k
        """
        self.last_query = query
        return self._docs[:k], self._scores[:k]


class DummyGenerator:
    """Mock generator for testing answer generation."""

    def __init__(self):
        """Initialize with empty call history."""
        self.calls = []

    def generate(self, question, contexts):
        """Mock generate method that records calls and returns dummy answers.

        Args:
            question: User question
            contexts: List of context strings

        Returns:
            Dummy answer string for testing
        """
        self.calls.append((question, contexts))
        return "dummy-answer-for:" + question


class DummyHistory:
    """Mock history storage for testing interaction persistence."""

    def __init__(self):
        """Initialize with empty save history."""
        self.saved = []

    def save(self, q, a, source_ids):
        """Mock save method that records all interactions.

        Args:
            q: Question text
            a: Answer text
            source_ids: List of source document IDs
        """
        self.saved.append((q, a, list(source_ids)))


# --- Test Cases ---


def test_rag_service_flow():
    """Test the complete RAG service workflow with successful document retrieval."""
    # --- Setup ---
    doc = Document(id=1, content="contenido relevante")
    retriever = DummyRetriever([doc], [0.85])
    generator = DummyGenerator()
    history = DummyHistory()

    rag = RagService(retriever, generator, history)

    # --- Execute ---
    resp = rag.ask("¿Qué es esto?", top_k=1)

    # --- Verify Response Structure ---
    assert resp["answer"].startswith("dummy-answer-for")
    assert resp["docs"] == [doc]
    assert resp["scores"] == [0.85]

    # --- Verify History Persistence ---
    assert history.saved == [("¿Qué es esto?", resp["answer"], [1])]


def test_rag_service_empty_docs():
    """Test RAG service behavior when no documents are found."""
    # --- Setup ---
    retriever = DummyRetriever([], [])
    generator = DummyGenerator()
    history = DummyHistory()
    rag = RagService(retriever, generator, history)

    # --- Execute ---
    resp = rag.ask("vacío", top_k=2)

    # --- Verify Empty Response Handling ---
    assert resp["answer"].startswith("No hay documentos indexados")
    assert resp["docs"] == []
    assert resp["scores"] == []

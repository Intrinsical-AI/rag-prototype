"""
RAG Prototype - Intrinsical-AI (c) 2025
Author: Pablo Pintor
License: MIT

Module: Pytest Configuration and Fixtures
Purpose: Provides shared test fixtures and configuration for the test suite.
         Includes database setup, mocking utilities, and test isolation helpers.
"""

from __future__ import annotations

from contextlib import suppress

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Import models to ensure they are registered with Base.metadata
from local_rag_backend.infrastructure.persistence.sqlalchemy import base as db_base, sql_


@pytest.fixture()
def in_memory_sqlite(monkeypatch):
    """Create an isolated in-memory SQLite database for testing.

    This fixture provides complete database isolation for each test by:
    - Creating a fresh in-memory SQLite database
    - Setting up all required tables from SQLAlchemy models
    - Patching global database connections to use the test database
    - Ensuring proper cleanup after test completion

    Args:
        monkeypatch: Pytest fixture for patching global objects

    Yields:
        sessionmaker: SQLAlchemy session factory for the test database

    Note:
        Uses StaticPool to ensure all connections share the same in-memory database
        and enables thread-safety for concurrent test execution.
    """
    # --- Database Setup ---
    # Use StaticPool so all sessions share the same in-memory database connection
    # and add check_same_thread=False for sqlite thread-safety in tests
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    # --- Schema Creation ---
    # Create tables from our Base (now that models are imported)
    db_base.Base.metadata.create_all(bind=engine)

    # --- Global Patching ---
    # Patch objects used in the code to use test database
    monkeypatch.setattr(db_base, "engine", engine)
    monkeypatch.setattr(db_base, "SessionLocal", TestingSessionLocal)
    # Also patch in the sql_ module so SqlDocumentStorage uses the test session
    monkeypatch.setattr(sql_, "SessionLocal", TestingSessionLocal)

    # --- Test Execution ---
    try:
        yield TestingSessionLocal
    finally:
        # --- Cleanup Phase ---
        # Ensure all sessions are closed
        with suppress(Exception):
            TestingSessionLocal.close_all_sessions()
        # Dispose engine to close underlying connection and avoid ResourceWarning
        engine.dispose()


# --- Test Utilities ---

class DummyFaissIndex:
    """Mock FAISS index implementation for testing purposes.

    This test double provides a simplified FAISS-like interface without
    requiring actual vector computations or file I/O operations.

    Attributes:
        index_path: Path where the real index would be stored
        id_map_path: Path where the ID mapping would be stored
        dim: Vector dimension (defaults to 4 for testing)
        id_map: List of document IDs that have been indexed
    """

    def __init__(self, index_path, id_map_path, dim=None):
        """Initialize the dummy FAISS index.

        Args:
            index_path: Mock path for the vector index file
            id_map_path: Mock path for the ID mapping file
            dim: Vector dimension (optional, defaults to 4)
        """
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.dim = dim or 4
        self.id_map = []

    def add_to_index(self, ids, vecs):
        """Mock method to add vectors to the index.

        Args:
            ids: List of document IDs to add
            vecs: List of vectors (ignored in mock)
        """
        self.id_map.extend(ids)

    def search(self, q, k):
        """Mock search method returning dummy results.

        Args:
            q: Query vector (ignored in mock)
            k: Number of results to return (ignored in mock)

        Returns:
            Tuple of (ids, scores) with dummy values
        """
        return ([0], [0.0])

    def similar(self, vector, k):
        """Mock similarity search returning dummy document-score pairs.

        Args:
            vector: Query vector (ignored in mock)
            k: Number of results to return (ignored in mock)

        Returns:
            List of (doc_id, similarity_score) tuples
        """
        if not self.id_map:
            return []
        return [(self.id_map[0], 0.5)] if self.id_map else []

    def save(self):
        """Mock save method (no-op for testing)."""
        pass

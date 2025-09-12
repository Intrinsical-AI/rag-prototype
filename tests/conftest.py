# ./conftest.py
import tempfile
from contextlib import suppress
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.infrastructure.persistence.sqlalchemy import base as db_base

# Import models to ensure they are registered with Base.metadata
from local_rag_backend.infrastructure.persistence.sqlalchemy import sql_


@pytest.fixture()
def in_memory_sqlite(monkeypatch):
    """
    Creates an in-memory SQLite database and patches global SessionLocal
    for all DAOs to use during tests.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    # Create tables registered with Base
    db_base.Base.metadata.create_all(bind=engine)

    # Patch objects used in code
    monkeypatch.setattr(db_base, "engine", engine)
    monkeypatch.setattr(db_base, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(sql_, "SessionLocal", TestingSessionLocal)

    try:
        yield TestingSessionLocal
    finally:
        with suppress(Exception):
            TestingSessionLocal.close_all()
        engine.dispose()


@pytest.fixture()
def temp_data_dir():
    """Provides a temporary directory for test data files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture()
def mock_embedder():
    """Provides a consistent mock embedder for tests."""
    embedder = Mock()
    embedder.dim = 384
    embedder.embed.return_value = [[0.1] * 384 for _ in range(10)]  # Batch of embeddings
    embedder.get_sentence_embedding_dimension.return_value = 384
    return embedder


@pytest.fixture()
def mock_llm_generator():
    """Provides a mock LLM generator for tests."""
    generator = Mock()
    generator.generate.return_value = "This is a test response based on the provided context."
    return generator


@pytest.fixture()
def sample_documents():
    """Provides sample documents for testing."""
    return [
        Document(id=1, content="Python is a programming language used for AI and web development."),
        Document(id=2, content="Machine learning algorithms can process large datasets efficiently."),
        Document(id=3, content="RAG combines retrieval and generation for better AI responses."),
        Document(id=4, content="Vector databases store high-dimensional embeddings for similarity search."),
        Document(id=5, content="FastAPI is a modern web framework for building APIs with Python."),
    ]


@pytest.fixture()
def mock_document_repo(sample_documents):
    """Provides a mock document repository with sample data."""
    repo = Mock()
    repo.get_all_documents.return_value = sample_documents
    repo.get.side_effect = lambda ids: [doc for doc in sample_documents if doc.id in ids]
    repo.store_documents.side_effect = lambda texts: list(range(len(sample_documents) + 1, len(sample_documents) + len(texts) + 1))
    return repo


@pytest.fixture()
def mock_vector_storage():
    """Provides a mock vector storage for tests."""
    storage = Mock()
    storage.similar.return_value = ([1, 2, 3], [0.9, 0.8, 0.7])
    storage.upsert.return_value = None
    return storage


class DummyFaissIndex:
    """Lightweight FAISS index mock for testing."""

    def __init__(self, index_path: str | Path, id_map_path: str | Path, dim: int = 384):
        self.index_path = Path(index_path)
        self.id_map_path = Path(id_map_path)
        self.dim = dim
        self.id_map: list[int] = []
        self._vectors: dict[int, list[float]] = {}

    def add_to_index(self, ids: list[int], vectors: list[list[float]]) -> None:
        """Add vectors to the mock index."""
        if vectors and len(vectors[0]) != self.dim:
            raise ValueError(f"FAISS dim mismatch: expected {self.dim}, got {len(vectors[0])}")

        for doc_id, vector in zip(ids, vectors, strict=True):
            if doc_id not in self.id_map:
                self.id_map.append(doc_id)
            self._vectors[doc_id] = vector

    def search(self, query_vector: list[float], k: int) -> tuple[list[int], list[float]]:
        """Mock search returning top-k results."""
        if not self.id_map:
            return [], []

        # Return first k IDs with mock distances
        result_ids = self.id_map[:k]
        result_distances = [0.1 * i for i in range(len(result_ids))]
        return result_ids, result_distances

    def save(self) -> None:
        """Mock save operation."""
        pass


@pytest.fixture()
def isolated_settings(monkeypatch, temp_data_dir):
    """Provides isolated settings for testing."""
    from local_rag_backend.settings import settings

    # Patch settings to use temp directory
    monkeypatch.setattr(settings, "data_dir", temp_data_dir)
    monkeypatch.setattr(settings, "sqlite_url", f"sqlite:///{temp_data_dir}/test.db")
    monkeypatch.setattr(settings, "index_path", str(temp_data_dir / "test.faiss"))
    monkeypatch.setattr(settings, "id_map_path", str(temp_data_dir / "test_id_map.pkl"))
    monkeypatch.setattr(settings, "faq_csv", str(temp_data_dir / "test_faq.csv"))

    return settings


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton state between tests."""
    try:
        from local_rag_backend.app import factory
        if hasattr(factory, "reset_rag_service"):
            factory.reset_rag_service()
    except ImportError:
        pass

    yield

    # Cleanup after test
    try:
        from local_rag_backend.app import factory
        if hasattr(factory, "reset_rag_service"):
            factory.reset_rag_service()
    except ImportError:
        pass

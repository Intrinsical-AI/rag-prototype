"""
End-to-end smoke tests for different retrieval modes.

These tests validate that the complete RAG pipeline works for both sparse (BM25)
and dense (FAISS) retrieval modes using temporary databases and indices.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from local_rag_backend.app.main import app
from local_rag_backend.core.domain.entities import Document
from local_rag_backend.infrastructure.persistence.faiss.index import FaissVectorStorage
from local_rag_backend.infrastructure.persistence.sqlalchemy.base import Base, engine
from local_rag_backend.infrastructure.persistence.sqlalchemy.crud import create_document
from local_rag_backend.infrastructure.persistence.sqlalchemy.models import DocumentModel
from local_rag_backend.infrastructure.retrievers.dense_faiss import DenseFaissRetriever
from local_rag_backend.infrastructure.retrievers.sparse_bm25 import SparseBM25Retriever
from local_rag_backend.settings import settings


@pytest.fixture
def temp_db():
    """Create temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        temp_db_path = tmp_db.name

    # Create temporary database URL
    temp_db_url = f"sqlite:///{temp_db_path}"

    # Patch settings to use temporary database
    with patch.object(settings, 'sqlite_url', temp_db_url):
        # Create tables in temporary database
        from sqlalchemy import create_engine
        temp_engine = create_engine(temp_db_url)
        Base.metadata.create_all(bind=temp_engine)

        yield temp_engine, temp_db_path

    # Cleanup
    Path(temp_db_path).unlink(missing_ok=True)


@pytest.fixture
def temp_faiss_index():
    """Create temporary FAISS index for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_index_path = Path(temp_dir) / "test_index.faiss"
        temp_id_map_path = Path(temp_dir) / "test_id_map.pkl"

        yield str(temp_index_path), str(temp_id_map_path)


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(id=1, content="Python is a programming language used for web development and data science."),
        Document(id=2, content="FastAPI is a modern web framework for building APIs with Python."),
        Document(id=3, content="Machine learning algorithms can process large datasets efficiently."),
        Document(id=4, content="Vector databases store high-dimensional embeddings for similarity search."),
        Document(id=5, content="Natural language processing helps computers understand human text."),
    ]


@pytest.mark.e2e
def test_sparse_retrieval_smoke(temp_db, sample_documents):
    """Smoke test for sparse (BM25) retrieval mode."""
    temp_engine, temp_db_path = temp_db

    # Insert sample documents into temporary database
    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(bind=temp_engine)

    with SessionLocal() as db_session:
        for doc in sample_documents:
            create_document(db=db_session, document=doc)
        db_session.commit()

    # Test sparse retrieval
    with patch.object(settings, 'sqlite_url', f"sqlite:///{temp_db_path}"):
        with patch.object(settings, 'retrieval_mode', 'sparse'):
            retriever = SparseBM25Retriever()

            # Test retrieval
            results = retriever.retrieve(query="Python programming", top_k=3)

            assert len(results) <= 3
            assert len(results) > 0

            # Verify results contain documents and scores
            for doc, score in results:
                assert isinstance(doc, Document)
                assert isinstance(score, (int, float))
                assert score >= 0.0
                assert doc.content is not None
                assert len(doc.content) > 0


@pytest.mark.e2e
def test_dense_retrieval_smoke(temp_db, temp_faiss_index, sample_documents):
    """Smoke test for dense (FAISS) retrieval mode."""
    temp_engine, temp_db_path = temp_db
    temp_index_path, temp_id_map_path = temp_faiss_index

    # Insert sample documents into temporary database
    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(bind=temp_engine)

    with SessionLocal() as db_session:
        for doc in sample_documents:
            create_document(db=db_session, document=doc)
        db_session.commit()

    # Create and populate FAISS index
    with patch.object(settings, 'sqlite_url', f"sqlite:///{temp_db_path}"):
        with patch.object(settings, 'index_path', temp_index_path):
            with patch.object(settings, 'id_map_path', temp_id_map_path):

                # Initialize vector storage and add documents
                vector_storage = FaissVectorStorage()

                # Add documents to FAISS index
                for doc in sample_documents:
                    vector_storage.add_document(doc)

                # Save the index
                vector_storage.save()

                # Test dense retrieval
                with patch.object(settings, 'retrieval_mode', 'dense'):
                    retriever = DenseFaissRetriever()

                    # Test retrieval
                    results = retriever.retrieve(query="Python web development", top_k=3)

                    assert len(results) <= 3
                    assert len(results) > 0

                    # Verify results contain documents and scores
                    for doc, score in results:
                        assert isinstance(doc, Document)
                        assert isinstance(score, (int, float))
                        assert score >= 0.0
                        assert doc.content is not None
                        assert len(doc.content) > 0


@pytest.mark.e2e
def test_api_health_endpoints():
    """Smoke test for health and readiness endpoints."""
    client = TestClient(app)

    # Test health endpoint
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

    # Test readiness endpoint
    response = client.get("/api/ready")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ready"


@pytest.mark.e2e
def test_api_openapi_documentation():
    """Smoke test for OpenAPI documentation enhancements."""
    client = TestClient(app)

    # Get OpenAPI schema
    response = client.get("/openapi.json")
    assert response.status_code == 200

    openapi_schema = response.json()

    # Verify health endpoints are documented
    assert "/api/health" in openapi_schema["paths"]
    assert "/api/ready" in openapi_schema["paths"]

    # Verify tags are present
    health_endpoint = openapi_schema["paths"]["/api/health"]["get"]
    assert "tags" in health_endpoint
    assert "Health" in health_endpoint["tags"]
    assert "summary" in health_endpoint

    # Verify RAG endpoints have proper tags
    ask_endpoint = openapi_schema["paths"]["/api/ask"]["post"]
    assert "tags" in ask_endpoint
    assert "RAG" in ask_endpoint["tags"]
    assert "summary" in ask_endpoint

    history_endpoint = openapi_schema["paths"]["/api/history"]["get"]
    assert "tags" in history_endpoint
    assert "RAG" in history_endpoint["tags"]
    assert "summary" in history_endpoint


@pytest.mark.e2e
def test_metrics_endpoint_when_monitoring_disabled():
    """Test metrics endpoint behavior when monitoring is disabled."""
    client = TestClient(app)

    # Test metrics endpoint (should work even when monitoring is disabled)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")

    # Should contain monitoring disabled message
    assert "Monitoring not enabled" in response.text


@pytest.mark.e2e
@patch.object(settings, 'enable_monitoring', True)
def test_metrics_endpoint_when_monitoring_enabled():
    """Test metrics endpoint behavior when monitoring is enabled."""
    # This test requires prometheus-client to be installed
    pytest.importorskip("prometheus_client")

    client = TestClient(app)

    # Make a request to generate some metrics
    client.get("/api/health")

    # Test metrics endpoint
    response = client.get("/metrics")
    assert response.status_code == 200

    # Should contain Prometheus metrics format
    content_type = response.headers.get("content-type", "")
    assert "text/plain" in content_type or "text/plain; version=0.0.4" in content_type

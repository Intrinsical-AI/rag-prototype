# ./conftest.py
from contextlib import suppress

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Import models to ensure they are registered with Base.metadata
from local_rag_backend.infrastructure.persistence.sqlalchemy import base as db_base, sql_


@pytest.fixture()
def in_memory_sqlite(monkeypatch):
    """
    Creates an in-memory SQLite database and patches SessionLocal globally
    so all DAOs use it during tests.
    """
    # Use StaticPool so all sessions share the same in-memory database connection
    # and add check_same_thread=False for sqlite thread-safety in tests
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    # Create tables that our Base (now that models are imported)
    db_base.Base.metadata.create_all(bind=engine)

    # Patch objects used in the code
    monkeypatch.setattr(db_base, "engine", engine)
    monkeypatch.setattr(db_base, "SessionLocal", TestingSessionLocal)
    # Also patch in the sql_ module so SqlDocumentStorage uses the test session
    monkeypatch.setattr(sql_, "SessionLocal", TestingSessionLocal)

    # Yield the session factory for tests that need it explicitly
    try:
        yield TestingSessionLocal
    finally:
        # Ensure all sessions are closed
        with suppress(Exception):
            TestingSessionLocal.close_all_sessions()
        # Dispose engine to close underlying connection and avoid ResourceWarning
        engine.dispose()


class DummyFaissIndex:
    def __init__(self, index_path, id_map_path, dim=None):  # <--- dim opcional
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.dim = 4
        self.id_map = []

    def add_to_index(self, ids, vecs):
        self.id_map.extend(ids)

    def search(self, q, k):
        return ([0], [0.0])

    def similar(self, vector, k):
        # Return (doc_id, similarity_score) pairs
        if not self.id_map:
            return []
        return [(self.id_map[0], 0.5)] if self.id_map else []

    def save(self):
        pass


@pytest.fixture()
async def asgi_client():
    """Async HTTP client against the ASGI app (avoids Starlette TestClient thread portal)."""
    import httpx

    from local_rag_backend.app.main import app

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

# tests/conftest.py
from contextlib import suppress

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Import models to ensure they are registered with Base.metadata
from local_rag_backend.infrastructure.persistence.sql import base as db_base, models as _models

_ = _models


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
    # Yield the session factory for tests that need it explicitly
    try:
        yield TestingSessionLocal
    finally:
        # Ensure all sessions are closed
        with suppress(Exception):
            TestingSessionLocal.close_all_sessions()
        # Dispose engine to close underlying connection and avoid ResourceWarning
        engine.dispose()


@pytest.fixture(autouse=True)
def reset_app_context_between_tests():
    """Ensure each test starts with a fresh app container/context."""
    from local_rag_backend.composition.factory import reset_app_context

    reset_app_context()
    try:
        yield
    finally:
        reset_app_context()


class DummyVectorIndex:
    def __init__(self, index_path, id_map_path, dim=None):  # <--- dim opcional
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.dim = 4
        self.id_map = []

    def add_to_index(self, ids, vecs):
        self.id_map.extend(ids)

    def delete_ids(self, ids):
        to_delete = set(ids)
        self.id_map = [x for x in self.id_map if x not in to_delete]
        return 0

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
async def asgi_client(in_memory_sqlite, monkeypatch, tmp_path):
    """Async HTTP client against the ASGI app using isolated in-memory SQLite."""
    _ = in_memory_sqlite
    import httpx

    from local_rag_backend.http.main import app
    from local_rag_backend.settings import settings

    # Give each test its own isolated data directory so that parallel
    # pytest-xdist workers don't share the mutation journal.  Without this,
    # the ASGI startup recovery (main.py:54) picks up PREPARED entries written
    # by other workers and injects phantom documents into this test's fresh
    # in-memory SQLite DB, causing spurious assertion failures.
    # settings.get_coordination_dir() returns data_dir.resolve() when data_dir
    # is absolute, so an absolute tmp_path fully isolates the journal.
    isolated_data_dir = tmp_path / "data"
    isolated_data_dir.mkdir()
    monkeypatch.setattr(settings, "data_dir", isolated_data_dir)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

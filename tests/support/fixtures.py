from contextlib import suppress

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from local_rag_backend.infrastructure.persistence.sql import base as db_base, models as _models

_ = _models


@pytest.fixture()
def in_memory_sqlite(monkeypatch):
    """Create an isolated in-memory SQLite DB and patch the global session factory."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session_local = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db_base.Base.metadata.create_all(bind=engine)

    monkeypatch.setattr(db_base, "engine", engine)
    monkeypatch.setattr(db_base, "SessionLocal", testing_session_local)
    try:
        yield testing_session_local
    finally:
        with suppress(Exception):
            testing_session_local.close_all_sessions()
        engine.dispose()


@pytest.fixture()
def reset_app_context():
    """Reset the composition context for tests that depend on a fresh container."""
    from local_rag_backend.composition.factory import reset_app_context as _reset_app_context

    _reset_app_context()
    try:
        yield
    finally:
        _reset_app_context()


@pytest.fixture()
async def asgi_client(reset_app_context, in_memory_sqlite, monkeypatch, tmp_path):
    """Async HTTP client against the ASGI app with isolated DB and coordination dir."""
    _ = (reset_app_context, in_memory_sqlite)
    import httpx

    from local_rag_backend.http.main import app
    from local_rag_backend.settings import settings

    isolated_data_dir = tmp_path / "data"
    isolated_data_dir.mkdir()
    monkeypatch.setattr(settings, "data_dir", isolated_data_dir)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

# tests/unit/app/test_api_history.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from local_rag_backend.app.dependencies import get_rag_service
from local_rag_backend.app.main import app
from local_rag_backend.infrastructure.persistence.sqlalchemy import models
from local_rag_backend.infrastructure.persistence.sqlalchemy.base import Base, get_db


@pytest.fixture()
def test_db_session_factory() -> sessionmaker:
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)
    try:
        yield session_factory
    finally:
        engine.dispose()


def setup_history_data(session: sessionmaker, monkeypatch) -> None:
    # Seed data
    try:
        session.add(models.QaHistory(question="Q1", answer="A1", source_ids=[1, 2]))
        session.add(models.QaHistory(question="Q2", answer="A2", source_ids=None))
        session.commit()
    finally:
        session.close()

    # Override dependencies BEFORE creating TestClient (for lifespan)
    app.dependency_overrides = {}

    # Patch the symbols used by FastAPI lifespan in app.main
    import local_rag_backend.app.main as app_main

    # Use an in-memory engine for lifespan DB init to avoid file side effects
    lp_engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=lp_engine)
    monkeypatch.setattr(app_main, "global_app_engine", lp_engine, raising=True)
    monkeypatch.setattr(app_main, "AppDeclarativeBase", Base, raising=True)
    monkeypatch.setattr(app_main, "get_rag_service", lambda *a, **k: object(), raising=True)
    # Also override the dependency injection for endpoints (not strictly necessary for /api/history)
    app.dependency_overrides[get_rag_service] = lambda: object()

    def _get_db_override():
        db = test_db_session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _get_db_override

    with TestClient(app) as client:
        resp = client.get("/api/history", params={"limit": 10, "offset": 0})
        assert resp.status_code == 200
        data = resp.json()

    assert isinstance(data, list) and len(data) == 2

    # Verify content and source_ids typing
    got = {item["question"]: item for item in data}
    assert got["Q1"]["answer"] == "A1"
    assert got["Q1"]["source_ids"] == [1, 2]
    assert got["Q2"]["answer"] == "A2"
    assert got["Q2"]["source_ids"] == []

    # Dispose the in-memory engine used by lifespan
    lp_engine.dispose()

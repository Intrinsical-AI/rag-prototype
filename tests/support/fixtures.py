import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from local_rag_backend.infrastructure.persistence.sql import base as db_base, models as _models

_ = _models


@pytest.fixture()
def in_memory_sqlite(monkeypatch, tmp_path):
    """Inject an isolated database into the entrypoint's application context."""
    from local_rag_backend.composition import factory
    from local_rag_backend.composition.container import AppContainer
    from local_rag_backend.composition.context import AppContext
    from local_rag_backend.settings import get_settings

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session_local = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db_base.Base.metadata.create_all(bind=engine)

    factory.reset_app_context()
    settings_obj = get_settings()
    monkeypatch.setattr(settings_obj, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings_obj, "index_path", str(tmp_path / "index.faiss"))
    monkeypatch.setattr(settings_obj, "id_map_path", str(tmp_path / "index.ids.json"))
    monkeypatch.setattr(settings_obj, "embedding_cache_db_path", tmp_path / "embeddings.sqlite")

    def build_context():
        container = AppContainer.from_settings(settings_obj, session_factory=testing_session_local)
        return AppContext(settings_obj=settings_obj, container=container)

    monkeypatch.setattr(factory, "_build_app_context", build_context)
    try:
        yield testing_session_local
    finally:
        factory.reset_app_context()
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
    from local_rag_backend.settings import get_settings

    isolated_data_dir = tmp_path / "data"
    isolated_data_dir.mkdir()
    monkeypatch.setattr(get_settings(), "data_dir", isolated_data_dir)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

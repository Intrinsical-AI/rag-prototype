import pytest

from local_rag_backend.http.main import app
from local_rag_backend.settings import settings


@pytest.mark.unit
async def test_public_bind_requires_api_key(monkeypatch, in_memory_sqlite, reset_app_context):
    _ = reset_app_context
    monkeypatch.setattr(settings, "app_host", "0.0.0.0", raising=False)  # noqa: S104
    monkeypatch.setattr(settings, "api_key", None, raising=False)
    monkeypatch.setattr(settings, "public_bind_requires_api_key", True, raising=False)

    with pytest.raises(RuntimeError, match="API key"):
        async with app.router.lifespan_context(app):
            pass


@pytest.mark.unit
async def test_public_bind_check_can_be_disabled(monkeypatch, in_memory_sqlite, reset_app_context):
    _ = reset_app_context
    monkeypatch.setattr(settings, "app_host", "0.0.0.0", raising=False)  # noqa: S104
    monkeypatch.setattr(settings, "api_key", None, raising=False)
    monkeypatch.setattr(settings, "public_bind_requires_api_key", False, raising=False)

    async with app.router.lifespan_context(app):
        pass

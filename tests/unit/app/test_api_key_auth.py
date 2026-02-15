import pytest

from local_rag_backend.settings import settings


@pytest.mark.unit
async def test_api_key_required_when_configured(asgi_client, in_memory_sqlite, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret", raising=False)

    r1 = await asgi_client.get("/api/health")
    assert r1.status_code == 401

    r2 = await asgi_client.get("/api/health", headers={"X-API-Key": "secret"})
    assert r2.status_code == 200
    assert r2.json().get("status") == "healthy"

    r3 = await asgi_client.get("/metrics")
    assert r3.status_code == 401

    r4 = await asgi_client.get("/metrics", headers={"X-API-Key": "secret"})
    assert r4.status_code == 200

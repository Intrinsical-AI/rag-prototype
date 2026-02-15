# tests/unit/app/test_api_router_endpoints_extra.py
from types import SimpleNamespace

from local_rag_backend.app import api_router as api
from local_rag_backend.settings import settings


async def test_health_db_failure(asgi_client, monkeypatch):
    class Boom(Exception):
        pass

    class DummyConn:
        def __enter__(self):
            raise Boom("db down")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(api.db_base, "engine", SimpleNamespace(connect=lambda: DummyConn()))
    r = await asgi_client.get("/api/health")
    assert r.status_code == 503
    assert "Database connection failed" in r.json()["detail"]


async def test_ready_not_ready_db_and_no_llm(asgi_client, monkeypatch):
    # Force DB failure and no providers
    class DummyConn:
        def __enter__(self):
            raise RuntimeError("no db")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(api.db_base, "engine", SimpleNamespace(connect=lambda: DummyConn()))
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", False, raising=False)

    async def _override():
        return None

    monkeypatch.setattr(api, "get_rag_service", _override, raising=True)
    r = await asgi_client.get("/api/ready")
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["status"] == "not_ready"
    checks = detail["checks"]
    assert checks.get("database", "").startswith("failed")
    assert (
        checks.get("rag_service", "").startswith("failed")
        or checks.get("rag_service") == "failed: not initialized"
    )
    assert checks.get("llm_providers", "").startswith("failed")


async def test_ollama_health_ok(asgi_client, monkeypatch):
    class Resp:
        def raise_for_status(self):
            return None

    monkeypatch.setattr(api.requests, "get", lambda *a, **k: Resp())
    r = await asgi_client.get("/api/health/ollama")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_ollama_health_fail(asgi_client, monkeypatch):
    class X(Exception):
        pass

    def boom(*a, **k):
        raise api.requests.exceptions.RequestException("oops")

    monkeypatch.setattr(api.requests, "get", boom)
    r = await asgi_client.get("/api/health/ollama")
    assert r.status_code == 503

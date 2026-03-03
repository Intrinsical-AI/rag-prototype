from local_rag_backend.http.routers import health as health_router
from local_rag_backend.settings import settings


async def test_health_db_failure(asgi_client, monkeypatch):
    class Boom(Exception):
        pass

    def _boom(*, diagnostics):
        raise Boom("db down")

    monkeypatch.setattr(health_router, "ping_database", _boom)
    r = await asgi_client.get("/healthz")
    assert r.status_code == 503
    assert "Database connection failed" in r.json()["detail"]


async def test_ready_not_ready_db_and_no_llm(asgi_client, monkeypatch):
    # Force DB failure and no providers
    def _db_failed(*, checks, diagnostics):
        checks["database"] = "failed: no db"
        return False

    monkeypatch.setattr(health_router, "check_database", _db_failed)
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", False, raising=False)

    async def _override():
        return None

    monkeypatch.setattr(health_router, "get_rag_service", _override, raising=True)
    r = await asgi_client.get("/readyz")
    assert r.status_code == 503
    detail = r.json()
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

    monkeypatch.setattr(health_router.httpx, "get", lambda *a, **k: Resp())
    r = await asgi_client.get("/healthz/ollama")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_ollama_health_fail(asgi_client, monkeypatch):
    def boom(*a, **k):
        raise health_router.httpx.HTTPError("oops")

    monkeypatch.setattr(health_router.httpx, "get", boom)
    r = await asgi_client.get("/healthz/ollama")
    assert r.status_code == 503

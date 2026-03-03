import httpx
import pytest
from starlette.requests import Request

from local_rag_backend.core.use_cases.errors import UnauthorizedError
from local_rag_backend.http.main import app
from local_rag_backend.http.security import require_api_key
from local_rag_backend.settings import settings


@pytest.mark.unit
async def test_api_key_required_when_configured(asgi_client, in_memory_sqlite, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret", raising=False)

    r1 = await asgi_client.get("/api/config")
    assert r1.status_code == 401

    r2 = await asgi_client.get("/api/config", headers={"X-API-Key": "secret"})
    assert r2.status_code == 200
    assert "retrieval_mode" in r2.json()

    r3 = await asgi_client.get("/metrics")
    assert r3.status_code == 401

    r4 = await asgi_client.get("/metrics", headers={"X-API-Key": "secret"})
    assert r4.status_code == 200


@pytest.mark.unit
async def test_non_local_requests_require_api_key_when_public_bind_guard_enabled(
    in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "api_key", None, raising=False)
    monkeypatch.setattr(settings, "public_bind_requires_api_key", True, raising=False)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(
            app=app, raise_app_exceptions=True, client=("203.0.113.7", 4242)
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/config")

    assert r.status_code == 401
    assert "non-local requests" in r.json()["detail"]


@pytest.mark.unit
async def test_non_local_requests_can_be_allowed_explicitly(in_memory_sqlite, monkeypatch):
    monkeypatch.setattr(settings, "api_key", None, raising=False)
    monkeypatch.setattr(settings, "public_bind_requires_api_key", False, raising=False)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(
            app=app, raise_app_exceptions=True, client=("203.0.113.9", 9000)
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/config")

    assert r.status_code == 200


@pytest.mark.unit
async def test_unknown_client_host_requires_api_key_when_public_bind_guard_enabled(monkeypatch):
    monkeypatch.setattr(settings, "api_key", None, raising=False)
    monkeypatch.setattr(settings, "public_bind_requires_api_key", True, raising=False)

    # Some ASGI deployments/tests may provide no `client` tuple in scope.
    request = Request({"type": "http", "method": "GET", "path": "/api/config", "headers": []})
    with pytest.raises(UnauthorizedError) as excinfo:
        await require_api_key(request)

    assert excinfo.value.status_code == 401
    assert "non-local requests" in str(excinfo.value.detail)


@pytest.mark.unit
async def test_forwarded_non_local_host_requires_api_key_even_when_client_is_local(
    in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "api_key", None, raising=False)
    monkeypatch.setattr(settings, "public_bind_requires_api_key", True, raising=False)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(
            app=app, raise_app_exceptions=True, client=("127.0.0.1", 4242)
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get(
                "/api/config",
                headers={"X-Forwarded-For": "203.0.113.7"},
            )

    assert r.status_code == 401
    assert "non-local requests" in r.json()["detail"]


@pytest.mark.unit
async def test_rfc7239_forwarded_non_local_host_requires_api_key_when_client_is_local(
    in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "api_key", None, raising=False)
    monkeypatch.setattr(settings, "public_bind_requires_api_key", True, raising=False)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(
            app=app, raise_app_exceptions=True, client=("127.0.0.1", 4242)
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get(
                "/api/config",
                headers={"Forwarded": "for=203.0.113.60;proto=https;by=203.0.113.43"},
            )

    assert r.status_code == 401
    assert "non-local requests" in r.json()["detail"]


@pytest.mark.unit
async def test_rfc7239_forwarded_unknown_requires_api_key_when_client_is_local(
    in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "api_key", None, raising=False)
    monkeypatch.setattr(settings, "public_bind_requires_api_key", True, raising=False)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(
            app=app, raise_app_exceptions=True, client=("127.0.0.1", 4242)
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get(
                "/api/config",
                headers={"Forwarded": "for=unknown;proto=https"},
            )

    assert r.status_code == 401
    assert "non-local requests" in r.json()["detail"]


@pytest.mark.unit
async def test_x_forwarded_for_blank_chain_requires_api_key_when_client_is_local(
    in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "api_key", None, raising=False)
    monkeypatch.setattr(settings, "public_bind_requires_api_key", True, raising=False)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(
            app=app, raise_app_exceptions=True, client=("127.0.0.1", 4242)
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get(
                "/api/config",
                headers={"X-Forwarded-For": " ,   "},
            )

    assert r.status_code == 401
    assert "non-local requests" in r.json()["detail"]


@pytest.mark.unit
async def test_public_probe_endpoints_do_not_require_api_key(
    asgi_client, in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "api_key", "secret", raising=False)
    r1 = await asgi_client.get("/healthz")
    assert r1.status_code == 200
    r2 = await asgi_client.get("/readyz")
    assert r2.status_code in (200, 503)

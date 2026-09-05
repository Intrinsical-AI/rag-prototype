import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import ASGITransport, AsyncClient

from local_rag_backend.http.main import _get_cors_allow_origins
from local_rag_backend.settings import Settings


@pytest.mark.unit
async def test_cors_preflight_does_not_enable_credentials(asgi_client, in_memory_sqlite):
    # Preflight request
    r = await asgi_client.options(
        "/api/config",
        headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    # By default (DEBUG=false + no CORS_ALLOW_ORIGINS), cross-origin requests are disallowed.
    assert r.status_code == 400
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}
    assert "access-control-allow-credentials" not in {k.lower() for k in r.headers}


@pytest.mark.unit
async def test_cors_preflight_allows_only_configured_production_origin():
    settings_obj = Settings(
        debug=False,
        cors_allow_origins=["https://allowed.example"],
    )
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_get_cors_allow_origins(settings_obj),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        allowed = await client.options(
            "/",
            headers={
                "Origin": "https://allowed.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        denied = await client.options(
            "/",
            headers={
                "Origin": "https://denied.example",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert allowed.headers["access-control-allow-origin"] == "https://allowed.example"
    assert "access-control-allow-origin" not in denied.headers


@pytest.mark.unit
def test_debug_cors_resolves_to_wildcard():
    assert _get_cors_allow_origins(Settings(debug=True)) == ["*"]

import pytest


@pytest.mark.unit
async def test_cors_preflight_does_not_enable_credentials(asgi_client, in_memory_sqlite):
    # Preflight request
    r = await asgi_client.options(
        "/api/health",
        headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Starlette typically returns 200 for handled preflights.
    assert r.status_code in (200, 204)
    assert r.headers.get("access-control-allow-origin") == "*"
    assert "access-control-allow-credentials" not in {k.lower() for k in r.headers}

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
    # By default (DEBUG=false + no CORS_ALLOW_ORIGINS), cross-origin requests are disallowed.
    assert r.status_code == 400
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}
    assert "access-control-allow-credentials" not in {k.lower() for k in r.headers}

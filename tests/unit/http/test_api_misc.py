# tests/unit/app/test_api_misc.py
from local_rag_backend.settings import settings


async def test_cors_preflight_options(asgi_client):
    r = await asgi_client.options(
        "/api/ask",
        headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    # By default (DEBUG=false + no CORS_ALLOW_ORIGINS), cross-origin requests are disallowed.
    assert r.status_code == 400
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


async def test_get_config_defaults(asgi_client, monkeypatch):
    # Ensure clean provider state
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", False, raising=False)

    r = await asgi_client.get("/api/config")
    assert r.status_code == 200
    data = r.json()
    assert set(
        [
            "search_backend",
            "retrieval_mode",
            "dual_candidate_k",
            "hybrid_alpha",
            "temperature",
            "max_tokens",
            "available_providers",
        ]
    ).issubset(data.keys())
    assert data["retrieval_mode"] == settings.retrieval_mode
    assert isinstance(data["available_providers"], list)


async def test_get_templates(asgi_client):
    r = await asgi_client.get("/api/templates")
    assert r.status_code == 200
    arr = r.json()
    names = {t["name"] for t in arr}
    assert {"default", "ollama", "concise", "detailed"}.issubset(names)

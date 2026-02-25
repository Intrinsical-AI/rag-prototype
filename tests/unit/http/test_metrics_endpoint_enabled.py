# tests/unit/app/test_metrics_endpoint_enabled.py

from local_rag_backend.http import middleware as mw
from local_rag_backend.settings import settings


async def test_metrics_endpoint_enabled_returns_metrics(asgi_client, monkeypatch):
    monkeypatch.setattr(settings, "enable_monitoring", True, raising=False)
    monkeypatch.setattr(mw, "PROMETHEUS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(mw, "CONTENT_TYPE_LATEST", "text/plain; version=0.0.4; charset=utf-8")
    monkeypatch.setattr(mw, "generate_latest", lambda: b"rag_queries_total 1\n", raising=False)

    r = await asgi_client.get("/metrics")
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("text/plain")
    assert "rag_queries_total" in r.text

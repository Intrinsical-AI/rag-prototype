# tests/unit/app/test_middleware_prometheus.py
import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from local_rag_backend.http import middleware as mw
from local_rag_backend.settings import settings


def test_get_metrics_enabled(monkeypatch):
    # Force metrics path
    monkeypatch.setattr(mw, "PROMETHEUS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(settings, "enable_monitoring", True, raising=False)

    class Dummy:
        @staticmethod
        def decode(arg):
            return "ok-metrics"

    monkeypatch.setattr(mw, "generate_latest", lambda: Dummy())
    monkeypatch.setattr(mw, "CONTENT_TYPE_LATEST", "text/plain; version=0.0.4; charset=utf-8")

    text, content_type = mw.get_metrics()
    assert text == "ok-metrics"
    assert content_type.startswith("text/plain")


async def test_middleware_dispatch_active(monkeypatch):
    # Build a small FastAPI app and attach the middleware explicitly
    monkeypatch.setattr(mw, "PROMETHEUS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(settings, "enable_monitoring", True, raising=False)

    app = FastAPI()
    app.add_middleware(mw.MetricsMiddleware)

    @app.get("/ping")
    async def ping():
        return PlainTextResponse("pong")

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/ping")
    assert r.status_code == 200
    assert r.text == "pong"


async def test_metrics_path_label_is_low_cardinality(monkeypatch):
    """
    Regression test: labeling metrics with the raw URL path can create unbounded
    series (e.g. /assets/<hash>.js or random 404 paths), causing memory DoS.
    """
    monkeypatch.setattr(mw, "PROMETHEUS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(settings, "enable_monitoring", True, raising=False)

    observed_paths: list[str] = []

    class DummyCounter:
        def labels(self, *, method, path, status_code):
            observed_paths.append(path)
            return self

        def inc(self, *_a, **_k):
            return None

    class DummyHistogram:
        def labels(self, *, method, path):
            observed_paths.append(path)
            return self

        def observe(self, *_a, **_k):
            return None

    monkeypatch.setattr(mw, "Counter", lambda *a, **k: DummyCounter(), raising=False)
    monkeypatch.setattr(mw, "Histogram", lambda *a, **k: DummyHistogram(), raising=False)

    app = FastAPI()
    app.add_middleware(mw.MetricsMiddleware)

    @app.get("/assets/{asset_path:path}")
    async def assets(asset_path: str):
        return PlainTextResponse("ok")

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Matched dynamic route should label using the route template, not the raw path.
        r1 = await client.get("/assets/a/b/c.js")
        assert r1.status_code == 200

        # Unmatched routes should not create a series per random path.
        r2 = await client.get("/does-not-exist-123")
        assert r2.status_code == 404

    assert "/assets/{asset_path:path}" in observed_paths
    assert "<unmatched>" in observed_paths

# tests/unit/app/test_middleware_prometheus.py
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient

from local_rag_backend.app import middleware as mw
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


def test_middleware_dispatch_active(monkeypatch):
    # Build a small FastAPI app and attach the middleware explicitly
    monkeypatch.setattr(mw, "PROMETHEUS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(settings, "enable_monitoring", True, raising=False)

    app = FastAPI()
    app.add_middleware(mw.MetricsMiddleware)

    @app.get("/ping")
    def ping():
        return PlainTextResponse("pong")

    client = TestClient(app)
    r = client.get("/ping")
    assert r.status_code == 200
    assert r.text == "pong"

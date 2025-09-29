# tests/unit/app/test_health_endpoints.py
from fastapi.testclient import TestClient

from local_rag_backend.app.dependencies import get_rag_service
from local_rag_backend.app.main import app
from local_rag_backend.settings import settings


class _DummyRag:
    def ask(self, question, top_k=3):
        return {"answer": "ok", "docs": [], "scores": []}


def test_health_endpoint_ok():
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json().get("status") == "healthy"


def test_ready_endpoint_503_without_llm(monkeypatch):
    # Ensure no providers are configured
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", False, raising=False)

    # Service should be available
    app.dependency_overrides[get_rag_service] = lambda: _DummyRag()

    client = TestClient(app)
    r = client.get("/api/ready")
    assert r.status_code == 503
    assert r.json()["detail"]["status"] == "not_ready"

    app.dependency_overrides.clear()


def test_ready_endpoint_200_with_openai(monkeypatch):
    # Configure OpenAI so at least one provider is available
    monkeypatch.setattr(settings, "openai_api_key", "DUMMY", raising=False)

    # Service should be available
    app.dependency_overrides[get_rag_service] = lambda: _DummyRag()

    client = TestClient(app)
    r = client.get("/api/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"

    app.dependency_overrides.clear()

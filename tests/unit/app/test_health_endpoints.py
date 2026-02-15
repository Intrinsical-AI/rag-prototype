# tests/unit/app/test_health_endpoints.py
from local_rag_backend.app import api_router as api
from local_rag_backend.settings import settings


class _DummyRag:
    def ask(self, question, top_k=3):
        return {"answer": "ok", "docs": [], "scores": []}


async def test_health_endpoint_ok(asgi_client):
    r = await asgi_client.get("/api/health")
    assert r.status_code == 200
    assert r.json().get("status") == "healthy"


async def test_ready_endpoint_503_without_llm(asgi_client, monkeypatch):
    # Ensure no providers are configured
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", False, raising=False)

    async def _override():
        return _DummyRag()

    monkeypatch.setattr(api, "get_rag_service", _override, raising=True)

    r = await asgi_client.get("/api/ready")
    assert r.status_code == 503
    assert r.json()["detail"]["status"] == "not_ready"


async def test_ready_endpoint_200_with_openai(asgi_client, monkeypatch):
    # Configure OpenAI so at least one provider is available
    monkeypatch.setattr(settings, "openai_api_key", "DUMMY", raising=False)

    async def _override():
        return _DummyRag()

    monkeypatch.setattr(api, "get_rag_service", _override, raising=True)

    r = await asgi_client.get("/api/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


async def test_ready_endpoint_503_when_dense_index_missing(asgi_client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "DUMMY", raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "index_path", str(tmp_path / "missing.faiss"), raising=False)
    monkeypatch.setattr(settings, "id_map_path", str(tmp_path / "missing.json"), raising=False)

    async def _override():
        return _DummyRag()

    monkeypatch.setattr(api, "get_rag_service", _override, raising=True)
    r = await asgi_client.get("/api/ready")
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["status"] == "not_ready"
    assert detail["checks"]["retrieval_index"].startswith("failed")

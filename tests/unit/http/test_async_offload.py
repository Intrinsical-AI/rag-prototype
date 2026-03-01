import threading

import pytest

from local_rag_backend.composition import adapters as composition_adapters
from local_rag_backend.http.dependencies import get_rag_service
from local_rag_backend.http.main import app
from local_rag_backend.settings import settings


@pytest.mark.unit
async def test_ask_runs_in_worker_thread(asgi_client, in_memory_sqlite):
    main_tid = threading.get_ident()
    seen: dict[str, int] = {}

    class DummyService:
        def ask(self, question: str, top_k: int):
            seen["tid"] = threading.get_ident()
            return {"answer": "ok", "docs": [], "scores": []}

    async def _override():
        return DummyService()

    app.dependency_overrides[get_rag_service] = _override
    try:
        r = await asgi_client.post("/api/ask", json={"question": "hi", "k": 1})
        assert r.status_code == 200
        assert seen["tid"] != main_tid
    finally:
        app.dependency_overrides.pop(get_rag_service, None)


@pytest.mark.unit
async def test_openrouter_generate_runs_in_worker_thread(asgi_client, monkeypatch):
    main_tid = threading.get_ident()
    seen: dict[str, int] = {}

    monkeypatch.setattr(settings, "openrouter_enabled", True, raising=False)
    monkeypatch.setattr(settings, "openrouter_api_key", "k", raising=False)

    class DummyUsage:
        prompt_tokens = 1
        completion_tokens = 2
        total_tokens = 3

    class DummyChoicesMsg:
        content = "hi"

    class DummyChoice:
        message = DummyChoicesMsg()

    class DummyResp:
        choices = [DummyChoice()]
        usage = DummyUsage()

    class DummyClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return DummyResp()

    def _fake_create_openai_client(**kwargs):
        seen["tid"] = threading.get_ident()
        return DummyClient()

    monkeypatch.setattr(
        composition_adapters,
        "create_openai_client",
        _fake_create_openai_client,
        raising=True,
    )

    r = await asgi_client.post(
        "/api/openrouter/generate",
        json={"system_instruction": "sys", "user_content": "hi"},
    )
    assert r.status_code == 200
    assert seen["tid"] != main_tid


@pytest.mark.unit
async def test_ollama_health_check_runs_in_worker_thread(asgi_client, monkeypatch):
    main_tid = threading.get_ident()
    seen: dict[str, int] = {}

    from local_rag_backend.http.routers import health as health_router

    class DummyResp:
        def raise_for_status(self):
            return None

    def _fake_get(url: str, timeout: int):
        seen["tid"] = threading.get_ident()
        return DummyResp()

    monkeypatch.setattr(health_router.httpx, "get", _fake_get)
    r = await asgi_client.get("/api/health/ollama")
    assert r.status_code == 200
    assert seen["tid"] != main_tid


@pytest.mark.unit
async def test_docs_dense_runs_heavy_path_in_worker_thread(
    asgi_client, in_memory_sqlite, monkeypatch
):
    main_tid = threading.get_ident()
    seen: dict[str, int] = {}

    from local_rag_backend.composition import factory

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)

    class DummyEmbedder:
        dim = 4

        def __init__(self, *a, **k):
            seen["tid"] = threading.get_ident()

        def embed(self, texts):
            return [[0.0, 0.0, 0.0, 0.0] for _ in texts]

    class DummyVec:
        def __init__(self, *a, **k):
            pass

        def upsert(self, ids, vectors):
            return None

        def apply_delta_atomic(self, *, delete_ids, upserts):
            return None

    monkeypatch.setattr(factory, "SentenceTransformerEmbedder", lambda **k: DummyEmbedder())
    monkeypatch.setattr(factory, "VectorStorage", lambda **k: DummyVec())

    r = await asgi_client.post("/api/docs", json={"texts": ["X"]})
    assert r.status_code == 200
    assert seen["tid"] != main_tid

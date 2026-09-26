import threading

import pytest
from support.container import override_container

from local_rag_backend.composition import adapters as composition_adapters
from local_rag_backend.core.domain.embeddings import EmbeddingIdentity
from local_rag_backend.http.routers import rag_router
from local_rag_backend.settings import get_settings

settings = get_settings()


@pytest.mark.unit
async def test_ask_runs_in_worker_thread(asgi_client, in_memory_sqlite, monkeypatch):
    main_tid = threading.get_ident()
    seen: dict[str, int] = {}

    class DummyService:
        def ask(self, question: str, top_k: int, *, filters=(), retrieval_mode="sparse"):
            _ = (question, top_k, filters, retrieval_mode)
            seen["tid"] = threading.get_ident()
            return {"answer": "ok", "docs": [], "scores": []}

    async def _override():
        return DummyService()

    monkeypatch.setattr(rag_router, "get_rag_service", _override, raising=True)
    r = await asgi_client.post("/api/ask", json={"question": "hi", "k": 1})
    assert r.status_code == 200
    assert seen["tid"] != main_tid


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
    r = await asgi_client.get("/healthz/ollama")
    assert r.status_code == 200
    assert seen["tid"] != main_tid


@pytest.mark.unit
async def test_docs_dense_runs_heavy_path_in_worker_thread(
    asgi_client, in_memory_sqlite, monkeypatch
):
    main_tid = threading.get_ident()
    seen: dict[str, int] = {}

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)

    class DummyEmbedder:
        dim = 4
        identity = EmbeddingIdentity(provider="openai", model="test", dimension=dim)

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

    override_container(monkeypatch, st_embedder_factory=lambda *a, **k: DummyEmbedder())
    override_container(monkeypatch, vector_repo_factory=lambda **k: DummyVec())

    r = await asgi_client.post("/api/docs/ingest", json={"texts": ["X"]})
    assert r.status_code == 200
    assert seen["tid"] != main_tid


@pytest.mark.parametrize("path", ["/api/docs/query", "/api/history"])
async def test_document_and_history_reads_run_in_worker_thread(asgi_client, monkeypatch, path):
    from local_rag_backend.http.routers import rag_router

    seen = []

    def read(*args, **kwargs):
        seen.append(threading.get_ident())
        return []

    if path == "/api/docs/query":
        monkeypatch.setattr(composition_adapters._RepoDocsReadPort, "query_docs", read)
        response = await asgi_client.post(path, json={})
    else:
        monkeypatch.setattr(rag_router, "list_history_entries_sync", read)
        response = await asgi_client.get(path)
    assert response.status_code == 200
    assert seen and seen[0] != threading.get_ident()


async def test_readiness_and_health_storage_checks_run_in_worker_threads(asgi_client, monkeypatch):
    from local_rag_backend.http.routers import health

    seen = []

    def check(**kwargs):
        seen.append(threading.get_ident())
        return True

    def counts(**kwargs):
        seen.append(threading.get_ident())
        return True, 0

    for name in (
        "check_database",
        "check_retrieval_index",
        "check_mutation_journal",
        "ping_database",
    ):
        monkeypatch.setattr(health, name, check)
    monkeypatch.setattr(health, "check_sql_counts", counts)
    await asgi_client.get("/readyz")
    response = await asgi_client.get("/healthz")
    assert response.status_code == 200
    assert len(seen) == 5
    assert all(tid != threading.get_ident() for tid in seen)

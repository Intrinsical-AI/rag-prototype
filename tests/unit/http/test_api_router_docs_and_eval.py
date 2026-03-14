# tests/unit/app/test_api_router_docs_and_eval.py

import numpy as np
import pytest

from local_rag_backend.composition import adapters as composition_adapters, factory
from local_rag_backend.http.routers import health as health_router
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.settings import settings


async def test_post_docs_sparse_and_list(asgi_client, in_memory_sqlite, monkeypatch):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    payload = {"texts": ["  First  ", "", "Second"]}
    r = await asgi_client.post("/api/docs", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2
    assert len(data["ids"]) == 2

    # Now list
    r2 = await asgi_client.get("/api/docs", params={"limit": 10, "offset": 0})
    assert r2.status_code == 200
    docs = r2.json()
    assert isinstance(docs, list)
    assert len(docs) >= 2
    assert set([d["id"] for d in docs]) >= set(data["ids"])  # ids contained


@pytest.mark.parametrize(
    "texts,expected",
    [
        (["  A  ", "", " B "], 2),
        (["á", "漢字", "   "], 2),
        (["dup", "dup", "  dup  "], 1),  # hash-based dedup (post-clean)
    ],
)
async def test_post_docs_sparse_various_inputs(
    asgi_client, in_memory_sqlite, monkeypatch, texts, expected
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    r = await asgi_client.post("/api/docs", json={"texts": texts})
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == expected


async def test_post_docs_dense_uses_etl(asgi_client, in_memory_sqlite, monkeypatch):
    # Dummy embedder/vector store
    class DummyEmbedder:
        dim = 4

        def embed(self, texts):
            return np.zeros((len(texts), self.dim), dtype="float32").tolist()

    class DummyVec:
        def __init__(self, *a, **k):
            self.calls = []

        def upsert(self, ids, vectors):
            self.calls.append((list(ids), list(vectors)))

        def apply_delta_atomic(self, *, delete_ids, upserts):
            ids = [str(doc_id) for doc_id, _ in upserts]
            vectors = [list(vec) for _, vec in upserts]
            self.calls.append((ids, vectors))
            assert list(delete_ids) == []

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(factory, "SentenceTransformerEmbedder", lambda **k: DummyEmbedder())
    dummy_vec = DummyVec()
    monkeypatch.setattr(factory, "VectorStorage", lambda **k: dummy_vec)

    payload = {"texts": ["X", "Y"]}
    r = await asgi_client.post("/api/docs", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2
    assert len(dummy_vec.calls) == 1
    ids_called, vectors_called = dummy_vec.calls[0]
    assert len(ids_called) == 2 and len(vectors_called) == 2


async def test_post_docs_sparse_dedup_is_idempotent(asgi_client, in_memory_sqlite, monkeypatch):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "ingest_chunker_version", "v1", raising=False)

    payload = {"texts": ["  DU P  ", "du p", "DU P"]}
    r1 = await asgi_client.post("/api/docs", json=payload)
    assert r1.status_code == 200
    ids1 = r1.json()["ids"]
    assert len(ids1) == 1

    docs1 = SqlDocumentStorage().get_all_documents()
    assert len(docs1) == 1

    r2 = await asgi_client.post("/api/docs", json=payload)
    assert r2.status_code == 200
    ids2 = r2.json()["ids"]
    assert ids2 == ids1

    docs2 = SqlDocumentStorage().get_all_documents()
    assert len(docs2) == 1


async def test_post_docs_sparse_chunker_version_change_inserts_new(
    asgi_client, in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "ingest_chunker_version", "v1", raising=False)
    payload = {"texts": ["hello world"]}

    r1 = await asgi_client.post("/api/docs", json=payload)
    assert r1.status_code == 200
    assert r1.json()["count"] == 1
    assert len(SqlDocumentStorage().get_all_documents()) == 1

    monkeypatch.setattr(settings, "ingest_chunker_version", "v2", raising=False)
    r2 = await asgi_client.post("/api/docs", json=payload)
    assert r2.status_code == 200
    assert r2.json()["count"] == 1
    assert len(SqlDocumentStorage().get_all_documents()) == 2


async def test_ask_eval_sparse_success(asgi_client, in_memory_sqlite, monkeypatch):
    # Seed DB with one doc
    store = SqlDocumentStorage()
    ids = store.store_documents(["hello world"])
    assert ids

    # Ensure OpenAI generator path but patch to avoid network
    class DummyGen:
        def __init__(self, *a, **k):
            pass

        def generate(self, question, contexts):
            return "ans"

    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(factory, "OpenAIGenerator", lambda **k: DummyGen())

    payload = {"question": "hello?", "config": {"retrieval_mode": "sparse", "k": 1}}
    r = await asgi_client.post("/api/ask_eval", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "ans"
    assert isinstance(data.get("sources", []), list)


async def test_ask_eval_rejects_unsafe_prompt_template(asgi_client, in_memory_sqlite, monkeypatch):
    # Ensure provider is "available" so we exercise config validation path deterministically.
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", False, raising=False)

    payload = {
        "question": "hi?",
        "config": {
            "retrieval_mode": "sparse",
            "k": 1,
            # Would be a memory-DoS vector with str.format; must be rejected.
            "prompt_template": "{question:100000000}",
        },
    }
    r = await asgi_client.post("/api/ask_eval", json=payload)
    assert r.status_code == 400
    assert "prompt_template" in r.json().get("detail", "")


async def test_ready_retrieval_index_present(asgi_client, in_memory_sqlite, tmp_path, monkeypatch):
    # Create a minimal valid on-disk index + id-map.
    from local_rag_backend.infrastructure.persistence.vector.storage import VectorStorage

    idx = tmp_path / "index.faiss"
    id_map = tmp_path / "id_map.json"
    monkeypatch.setattr(settings, "openai_api_key", "x", raising=False)
    VectorStorage(str(idx), str(id_map), dim=4).rebuild([], [])
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "index_path", str(idx), raising=False)
    monkeypatch.setattr(settings, "id_map_path", str(id_map), raising=False)

    # Provide deps
    class _Dummy:
        pass

    async def _override():
        return _Dummy()

    monkeypatch.setattr(health_router, "get_rag_service", _override, raising=True)

    r = await asgi_client.get("/readyz")
    assert r.status_code == 200
    checks = r.json()["checks"]
    assert checks.get("retrieval_index") == "ok"


async def test_openrouter_generate_success(asgi_client, monkeypatch):
    monkeypatch.setattr(settings, "openrouter_enabled", True, raising=False)
    monkeypatch.setattr(settings, "openrouter_api_key", "k", raising=False)
    monkeypatch.setattr(settings, "openai_request_timeout", 19, raising=False)
    captured: dict[str, object] = {}

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
        captured.update(kwargs)
        return DummyClient()

    monkeypatch.setattr(
        composition_adapters,
        "create_openai_client",
        _fake_create_openai_client,
        raising=True,
    )

    r = await asgi_client.post(
        "/api/openrouter/generate",
        json={
            "model": None,
            "system_instruction": "sys",
            "user_content": "hi",
            "temperature": 0.5,
            "max_tokens": 10,
            "top_p": 1.0,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["text"] == "hi"
    assert data["usage"]["prompt_tokens"] == 1
    assert captured.get("timeout") == 19


async def test_openrouter_generate_malformed_response_is_502(asgi_client, monkeypatch):
    monkeypatch.setattr(settings, "openrouter_enabled", True, raising=False)
    monkeypatch.setattr(settings, "openrouter_api_key", "k", raising=False)

    class DummyResp:
        choices: list[object] = []
        usage = None

    class DummyClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return DummyResp()

    monkeypatch.setattr(
        composition_adapters,
        "create_openai_client",
        lambda **kwargs: DummyClient(),
        raising=True,
    )

    r = await asgi_client.post(
        "/api/openrouter/generate",
        json={
            "model": None,
            "system_instruction": "sys",
            "user_content": "hi",
            "temperature": 0.5,
            "max_tokens": 10,
            "top_p": 1.0,
        },
    )
    assert r.status_code == 502
    assert "malformed response" in r.json()["detail"]

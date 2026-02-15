# tests/unit/app/test_api_router_docs_and_eval.py

import numpy as np
import pytest

from local_rag_backend.app import api_router as api
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage
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
        (["dup", "dup", "  dup  "], 3),  # API stores all non-empty entries; no dedup here
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

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(api, "SentenceTransformerEmbedder", lambda **k: DummyEmbedder())
    dummy_vec = DummyVec()
    monkeypatch.setattr(api, "FaissVectorStorage", lambda **k: dummy_vec)

    payload = {"texts": ["X", "Y"]}
    r = await asgi_client.post("/api/docs", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2
    assert len(dummy_vec.calls) == 1
    ids_called, vectors_called = dummy_vec.calls[0]
    assert len(ids_called) == 2 and len(vectors_called) == 2


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
    monkeypatch.setattr(api, "OpenAIGenerator", lambda **k: DummyGen())

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


async def test_ready_retrieval_index_present(asgi_client, tmp_path, monkeypatch):
    # Create dummy index file
    idx = tmp_path / "index.faiss"
    idx.write_text("")
    id_map = tmp_path / "id_map.json"
    id_map.write_bytes(b"")  # only need to exist for readiness
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "index_path", str(idx), raising=False)
    monkeypatch.setattr(settings, "id_map_path", str(id_map), raising=False)

    # Provide deps
    class _Dummy:
        pass

    async def _override():
        return _Dummy()

    monkeypatch.setattr(api, "get_rag_service", _override, raising=True)
    monkeypatch.setattr(settings, "openai_api_key", "x", raising=False)

    r = await asgi_client.get("/api/ready")
    assert r.status_code == 200
    checks = r.json()["checks"]
    assert checks.get("retrieval_index") == "ok"


async def test_openrouter_generate_success(asgi_client, monkeypatch):
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
        def __init__(self, *args, **kwargs):
            pass

        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return DummyResp()

    monkeypatch.setattr(api, "OpenAI", DummyClient)

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

from __future__ import annotations

from local_rag_backend.composition import factory
from local_rag_backend.core.domain.entities import Document
from local_rag_backend.http.routers import health as health_router, rag_router
from local_rag_backend.infrastructure.persistence.sql import HistorySqlStorage
from local_rag_backend.settings import settings


async def test_golden_f1_f2_docs_mutation_ingest_and_list(
    asgi_client,
    in_memory_sqlite,
    monkeypatch,
):
    _ = in_memory_sqlite
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    r1 = await asgi_client.post(
        "/api/docs/mutate",
        json={
            "upserts": [
                {"external_id": "gold:a", "content": "Alpha"},
                {"external_id": "gold:b", "content": "Beta"},
            ]
        },
    )
    assert r1.status_code == 200
    p1 = r1.json()
    assert p1["inserted"] == 2
    assert set(p1.keys()) >= {
        "op_id",
        "inserted",
        "updated",
        "unchanged",
        "deleted_sql",
        "deleted_index",
        "tombstoned",
        "missing_external_ids",
        "index_rebuilt",
        "index_doc_count",
        "results",
    }

    r2 = await asgi_client.post(
        "/api/docs/mutate",
        json={"delete_external_ids": ["gold:a"]},
    )
    assert r2.status_code == 200
    p2 = r2.json()
    assert p2["deleted_sql"] == 1
    assert p2["tombstoned"] == 1

    r3 = await asgi_client.post(
        "/api/docs/mutate",
        json={"upserts": [{"external_id": "gold:a", "content": "Alpha (blocked)"}]},
    )
    assert r3.status_code == 400

    r4 = await asgi_client.post("/api/docs/ingest", json={"texts": ["Uno", "Dos"]})
    assert r4.status_code == 200
    p4 = r4.json()
    assert p4["count"] == 2
    assert len(p4["ids"]) == 2

    r5 = await asgi_client.post("/api/docs/query", json={"limit": 100, "offset": 0, "filters": []})
    assert r5.status_code == 200
    docs = r5.json()
    assert isinstance(docs, list)
    assert len(docs) >= 3
    assert all(
        set(item.keys()) >= {"id", "content", "external_id", "source_id", "metadata"}
        for item in docs
    )


async def test_golden_f3_query_eval_and_history_contract(
    asgi_client,
    in_memory_sqlite,
    monkeypatch,
):
    _ = in_memory_sqlite
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    class _DummyRagService:
        def ask(
            self,
            question: str,
            top_k: int = 3,
            *,
            filters=(),
            retrieval_mode: str = "sparse",
        ) -> dict[str, object]:
            _ = (filters, retrieval_mode)
            return {
                "answer": f"echo:{question}",
                "docs": [Document(id="doc:1", content="ctx")],
                "scores": [0.9],
            }

    async def _override_rag_service() -> _DummyRagService:
        return _DummyRagService()

    class _DummyRagRuntimeFactory:
        def run_ask_eval(self, *, question: str, cfg: object) -> dict[str, object]:
            _ = cfg
            return {
                "answer": f"eval:{question}",
                "docs": [Document(id="doc:2", content="ctx-eval")],
                "scores": [0.8],
            }

    monkeypatch.setattr(rag_router, "get_rag_service", _override_rag_service, raising=True)
    container = factory.get_app_context().container
    monkeypatch.setattr(
        container,
        "build_rag_runtime_factory",
        lambda: _DummyRagRuntimeFactory(),
        raising=True,
    )
    r1 = await asgi_client.post("/api/ask", json={"question": "hello", "k": 1})
    assert r1.status_code == 200
    p1 = r1.json()
    assert p1["answer"] == "echo:hello"
    assert isinstance(p1["sources"], list) and len(p1["sources"]) == 1
    assert set(p1["sources"][0].keys()) >= {"document", "score"}

    r2 = await asgi_client.post(
        "/api/ask_eval",
        json={
            "question": "hello",
            "config": {
                "retrieval_mode": "sparse",
                "k": 1,
            },
        },
    )
    assert r2.status_code == 200
    p2 = r2.json()
    assert p2["answer"] == "eval:hello"
    assert isinstance(p2.get("latency_ms"), int)
    assert isinstance(p2["sources"], list) and len(p2["sources"]) == 1

    HistorySqlStorage().save("q-golden", "a-golden", ["doc:1"])
    r3 = await asgi_client.get("/api/history?limit=1")
    assert r3.status_code == 200
    rows = r3.json()
    assert isinstance(rows, list) and rows
    assert set(rows[0].keys()) >= {"id", "question", "answer", "created_at", "source_ids"}


async def test_golden_f4_f5_rebuild_and_readiness(
    asgi_client,
    in_memory_sqlite,
    tmp_path,
    monkeypatch,
):
    _ = in_memory_sqlite
    monkeypatch.setattr(settings, "openai_api_key", "DUMMY", raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "index_path", str(tmp_path / "golden-index.faiss"), raising=False)
    monkeypatch.setattr(
        settings, "id_map_path", str(tmp_path / "golden-id-map.json"), raising=False
    )

    class _DummyEmbedder:
        dim = 4

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0, 0.0, 0.0, 0.0] for _ in texts]

    async def _ready_service_override() -> object:
        return object()

    monkeypatch.setattr(factory, "OpenAIEmbedder", lambda *a, **k: _DummyEmbedder(), raising=True)
    monkeypatch.setattr(health_router, "get_rag_service", _ready_service_override, raising=True)
    factory.reset_app_context()

    r0 = await asgi_client.post(
        "/api/docs/mutate",
        json={"upserts": [{"external_id": "gold:rebuild", "content": "for-index"}]},
    )
    assert r0.status_code == 200

    r1 = await asgi_client.post("/api/index/rebuild")
    assert r1.status_code == 200
    assert int(r1.json()["indexed"]) >= 1

    r2 = await asgi_client.get("/healthz")
    assert r2.status_code == 200
    assert r2.json().get("status") == "healthy"

    r3 = await asgi_client.get("/readyz")
    assert r3.status_code == 200
    assert r3.json().get("status") == "ready"

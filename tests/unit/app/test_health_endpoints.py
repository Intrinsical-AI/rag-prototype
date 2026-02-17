# tests/unit/app/test_health_endpoints.py
from local_rag_backend.app.routers import health as health_router
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

    monkeypatch.setattr(health_router, "get_rag_service", _override, raising=True)

    r = await asgi_client.get("/api/ready")
    assert r.status_code == 503
    assert r.json()["detail"]["status"] == "not_ready"


async def test_ready_endpoint_200_with_openai(asgi_client, monkeypatch):
    # Configure OpenAI so at least one provider is available
    monkeypatch.setattr(settings, "openai_api_key", "DUMMY", raising=False)

    async def _override():
        return _DummyRag()

    monkeypatch.setattr(health_router, "get_rag_service", _override, raising=True)

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

    monkeypatch.setattr(health_router, "get_rag_service", _override, raising=True)
    r = await asgi_client.get("/api/ready")
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["status"] == "not_ready"
    assert detail["checks"]["retrieval_index"].startswith("failed")


async def test_ready_endpoint_503_when_dense_index_drifts_from_sql(
    asgi_client, in_memory_sqlite, tmp_path, monkeypatch
):
    from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
    from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

    monkeypatch.setattr(settings, "openai_api_key", "DUMMY", raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)

    idx = tmp_path / "index.faiss"
    id_map = tmp_path / "id_map.json"
    FaissVectorStorage(str(idx), str(id_map), dim=4).rebuild([], [])
    monkeypatch.setattr(settings, "index_path", str(idx), raising=False)
    monkeypatch.setattr(settings, "id_map_path", str(id_map), raising=False)

    # Seed DB with one document, but keep the index empty => drift.
    SqlDocumentStorage().store_documents(["hello"])

    async def _override():
        return _DummyRag()

    monkeypatch.setattr(health_router, "get_rag_service", _override, raising=True)
    r = await asgi_client.get("/api/ready")
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["status"] == "not_ready"
    assert detail["checks"]["retrieval_index"].startswith("failed: drift")


async def test_ready_endpoint_503_when_dense_index_id_set_mismatch(
    asgi_client, in_memory_sqlite, tmp_path, monkeypatch
):
    """
    Counts can match while the actual ID set differs (stale vectors + missing docs).
    /api/ready should catch this for small corpora and return actionable drift details.
    """
    from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
    from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

    monkeypatch.setattr(settings, "openai_api_key", "DUMMY", raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)

    # DB has ids {1,2}
    SqlDocumentStorage().store_documents(["d1", "d2"])

    # Index has ids {1,3} (same count, different set)
    idx = tmp_path / "index.faiss"
    id_map = tmp_path / "id_map.json"
    FaissVectorStorage(str(idx), str(id_map), dim=4).rebuild(
        [1, 3], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    monkeypatch.setattr(settings, "index_path", str(idx), raising=False)
    monkeypatch.setattr(settings, "id_map_path", str(id_map), raising=False)

    async def _override():
        return _DummyRag()

    monkeypatch.setattr(health_router, "get_rag_service", _override, raising=True)
    r = await asgi_client.get("/api/ready")
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["status"] == "not_ready"
    assert detail["checks"]["retrieval_index"].startswith(
        "failed: drift detected (ID set mismatch)"
    )
    drift = detail["checks"]["retrieval_index_drift"]
    assert drift["stale_count"] == 1
    assert drift["missing_count"] == 1


async def test_ready_endpoint_503_when_dense_id_map_is_corrupt(
    asgi_client, in_memory_sqlite, tmp_path, monkeypatch
):
    from local_rag_backend.infrastructure.persistence.faiss.index import FaissIndex

    monkeypatch.setattr(settings, "openai_api_key", "DUMMY", raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)

    idx = tmp_path / "index.faiss"
    id_map = tmp_path / "id_map.json"
    FaissIndex(idx, id_map, dim=4).rebuild([], [])
    id_map.write_bytes(b"")  # invalid/empty format
    monkeypatch.setattr(settings, "index_path", str(idx), raising=False)
    monkeypatch.setattr(settings, "id_map_path", str(id_map), raising=False)

    async def _override():
        return _DummyRag()

    monkeypatch.setattr(health_router, "get_rag_service", _override, raising=True)
    r = await asgi_client.get("/api/ready")
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["status"] == "not_ready"
    assert detail["checks"]["retrieval_index"].startswith("failed: corrupt")


async def test_ready_endpoint_503_when_dense_manifest_missing(
    asgi_client, in_memory_sqlite, tmp_path, monkeypatch
):
    from local_rag_backend.infrastructure.persistence.faiss.index import FaissIndex

    monkeypatch.setattr(settings, "openai_api_key", "DUMMY", raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)

    idx = tmp_path / "index.faiss"
    id_map = tmp_path / "id_map.json"
    # Create index files but deliberately do NOT create a manifest.
    FaissIndex(idx, id_map, dim=4).rebuild([], [])
    monkeypatch.setattr(settings, "index_path", str(idx), raising=False)
    monkeypatch.setattr(settings, "id_map_path", str(id_map), raising=False)

    async def _override():
        return _DummyRag()

    monkeypatch.setattr(health_router, "get_rag_service", _override, raising=True)

    r = await asgi_client.get("/api/ready")
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["status"] == "not_ready"
    assert "manifest" in detail["checks"]["retrieval_index_stats"].get("error", "")


async def test_ready_endpoint_503_when_dense_manifest_mismatch_embedding_model(
    asgi_client, in_memory_sqlite, tmp_path, monkeypatch
):
    from local_rag_backend.infrastructure.persistence.faiss.index import FaissIndex
    from local_rag_backend.infrastructure.persistence.faiss.manifest import (
        build_expected_manifest_config,
        build_manifest,
        manifest_path_for,
        write_manifest,
    )

    monkeypatch.setattr(settings, "openai_api_key", "DUMMY", raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)

    idx = tmp_path / "index.faiss"
    id_map = tmp_path / "id_map.json"
    FaissIndex(idx, id_map, dim=4).rebuild([], [])
    monkeypatch.setattr(settings, "index_path", str(idx), raising=False)
    monkeypatch.setattr(settings, "id_map_path", str(id_map), raising=False)

    expected = build_expected_manifest_config(
        embedding_backend="openai",
        embedding_model=settings.openai_embedding_model,
        chunker_strategy=settings.ingest_chunk_strategy,
        chunker_version=settings.ingest_chunker_version,
    )
    bad = dict(expected)
    bad["embedding_model"] = "some-other-embedding-model"
    write_manifest(
        manifest_path_for(idx),
        build_manifest(
            expected=bad,
            dimension=4,
            index_backend="numpy",  # backend mismatch is reported, but model mismatch must fail readiness
        ),
    )

    async def _override():
        return _DummyRag()

    monkeypatch.setattr(health_router, "get_rag_service", _override, raising=True)

    r = await asgi_client.get("/api/ready")
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["status"] == "not_ready"
    stats = detail["checks"]["retrieval_index_stats"]
    assert stats["status"] == "drift"
    assert any(m.get("key") == "embedding_model" for m in stats.get("manifest_mismatches", []))


async def test_rebuild_index_rewrites_manifest_and_ready_becomes_ok(
    asgi_client, in_memory_sqlite, tmp_path, monkeypatch
):
    from pathlib import Path

    from local_rag_backend.infrastructure.persistence.faiss.index import FaissIndex
    from local_rag_backend.infrastructure.persistence.faiss.manifest import (
        build_expected_manifest_config,
        build_manifest,
        manifest_path_for,
        write_manifest,
    )

    monkeypatch.setattr(settings, "openai_api_key", "DUMMY", raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "index_path", str(tmp_path / "index.faiss"), raising=False)
    monkeypatch.setattr(settings, "id_map_path", str(tmp_path / "id_map.json"), raising=False)

    # Create an empty index + a bad manifest.
    FaissIndex(Path(settings.index_path), Path(settings.id_map_path), dim=4).rebuild([], [])
    expected = build_expected_manifest_config(
        embedding_backend="openai",
        embedding_model=settings.openai_embedding_model,
        chunker_strategy=settings.ingest_chunk_strategy,
        chunker_version=settings.ingest_chunker_version,
    )
    bad = dict(expected)
    bad["chunker_version"] = "old"
    write_manifest(
        manifest_path_for(settings.index_path),
        build_manifest(expected=bad, dimension=4, index_backend="numpy"),
    )

    async def _override():
        return _DummyRag()

    monkeypatch.setattr(health_router, "get_rag_service", _override, raising=True)

    # Not ready due to manifest drift.
    r1 = await asgi_client.get("/api/ready")
    assert r1.status_code == 503

    # Rebuild should rewrite manifest to match current settings.
    rr = await asgi_client.post("/api/index/rebuild")
    assert rr.status_code == 200

    r2 = await asgi_client.get("/api/ready")
    assert r2.status_code == 200

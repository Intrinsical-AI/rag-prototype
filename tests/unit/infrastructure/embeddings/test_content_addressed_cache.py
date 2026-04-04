from __future__ import annotations

import json
from pathlib import Path

from local_rag_backend.infrastructure.embeddings.cached import (
    ContentAddressedCachingEmbedder,
    resolve_embedding_cache_db_path,
    resolve_embedding_model_key,
)
from local_rag_backend.infrastructure.observability.perf import (
    dump_perf_metrics_if_configured,
    reset_perf_metrics,
    snapshot_perf_metrics,
)


class _Embedder:
    dim = 2
    model_name = "toy-model"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts):
        texts_list = [str(text) for text in texts]
        self.calls.append(texts_list)
        return [[float(len(text)), float(len(text) + 1)] for text in texts_list]


def test_content_addressed_cache_reuses_embeddings_across_calls(tmp_path: Path) -> None:
    reset_perf_metrics()
    base = _Embedder()
    cached = ContentAddressedCachingEmbedder(base=base, cache_db_path=tmp_path / "cache.sqlite3")

    first = cached.embed(["alpha", "beta", "alpha"])
    second = cached.embed(["beta", "gamma"])

    assert base.calls == [["alpha", "beta"], ["gamma"]]
    assert first[0] == first[2]
    assert second[0] == first[1]

    metrics = snapshot_perf_metrics()["embedding_cache"]
    assert metrics["hits"] >= 1
    assert metrics["misses"] == 3
    assert metrics["embedded_vectors"] == 3
    assert metrics["estimated_saved_embed_seconds"] >= 0.0


def test_perf_metrics_dump_aggregates_process_payloads(monkeypatch, tmp_path: Path) -> None:
    reset_perf_metrics()
    base = _Embedder()
    cached = ContentAddressedCachingEmbedder(base=base, cache_db_path=tmp_path / "cache.sqlite3")
    cached.embed(["alpha", "alpha"])

    out = tmp_path / "perf.json"
    monkeypatch.setenv("RAG_PERF_METRICS_OUT", str(out))
    dump_perf_metrics_if_configured()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["processes"]
    assert payload["summary"]["embedding_cache"]["lookups"] >= 1
    assert payload["summary"]["embedding_cache"]["embedded_vectors"] >= 1


def test_resolve_embedding_model_key_is_explicit() -> None:
    assert resolve_embedding_model_key(_Embedder()) == "sentence_transformers:toy-model:2"


def test_resolve_embedding_model_key_detects_openai_like_embedder() -> None:
    class _OpenAIEmbedder:
        dim = 3
        model = "text-embedding-3-small"
        client = object()

    assert resolve_embedding_model_key(_OpenAIEmbedder()) == "openai:text-embedding-3-small:3"


def test_resolve_embedding_cache_db_path_honors_override(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "custom-cache.sqlite3"
    monkeypatch.setenv("RAG_EMBEDDING_CACHE_DB", str(override))
    assert resolve_embedding_cache_db_path(data_dir=tmp_path / "data") == override

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from local_rag_backend.core.domain.embeddings import EmbeddingIdentity
from local_rag_backend.infrastructure.embeddings.cached import (
    ContentAddressedCachingEmbedder,
    resolve_embedding_cache_db_path,
)
from local_rag_backend.infrastructure.observability.perf import (
    dump_perf_metrics_if_configured,
    reset_perf_metrics,
    snapshot_perf_metrics,
)


class _Embedder:
    dim = 2
    model_name = "toy-model"
    identity = EmbeddingIdentity(provider="sentence_transformers", model="toy-model", dimension=2)

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
    dump_perf_metrics_if_configured(out)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["processes"]
    assert payload["summary"]["embedding_cache"]["lookups"] >= 1
    assert payload["summary"]["embedding_cache"]["embedded_vectors"] >= 1


def test_cache_separates_real_synthetic_and_implementation_versions(tmp_path: Path) -> None:
    synthetic, real, revised = _Embedder(), _Embedder(), _Embedder()
    synthetic.identity = replace(synthetic.identity, synthetic=True)
    revised.identity = replace(revised.identity, implementation_version="2")
    cached = [
        ContentAddressedCachingEmbedder(base=base, cache_db_path=tmp_path / "cache.sqlite3")
        for base in (synthetic, real, revised)
    ]
    for embedder in cached:
        embedder.embed(["same text"])
        embedder.embed(["same text"])
    assert len({embedder.model_key for embedder in cached}) == 3
    assert all(base.calls == [["same text"]] for base in (synthetic, real, revised))


def test_resolve_embedding_cache_db_path_honors_override(tmp_path: Path) -> None:
    override = tmp_path / "custom-cache.sqlite3"
    assert (
        resolve_embedding_cache_db_path(
            data_dir=tmp_path / "data",
            configured_path=override,
        )
        == override
    )

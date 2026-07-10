from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

import local_rag_backend.integrations.embeddings as embedding_api
from local_rag_backend.integrations.embeddings import (
    EmbeddingLimits,
    EmbeddingsBackendUnavailableError,
    EmbeddingService,
    _factory as embedding_factory,
    create_embedding_service,
)
from local_rag_backend.integrations.embeddings._service import _ConfiguredEmbeddingService
from local_rag_backend.settings import Settings


def _write_synthetic_config(
    tmp_path: Path,
    *,
    cache_enabled: bool = True,
    dimension: int = 6,
) -> Path:
    config_dir = tmp_path / "runtime"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "openai_api_key: null",
                "st_embedding_model: synthetic-test-model",
                "synthetic_embeddings: true",
                f"synthetic_embedding_dim: {dimension}",
                "synthetic_embedding_fail_rate: 0.0",
                "synthetic_embedding_jitter_min_ms: 0.0",
                "synthetic_embedding_jitter_max_ms: 0.0",
                "data_dir: data",
                "embedding_cache_db_path: data/embedding-cache.sqlite3",
                f"disable_embedding_cache: {str(not cache_enabled).lower()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def test_public_api_exports_only_documented_names() -> None:
    assert embedding_api.__all__ == [
        "DEFAULT_EMBEDDING_LIMITS",
        "EmbeddingLimits",
        "EmbeddingService",
        "EmbeddingStatus",
        "EmbeddingsBackendUnavailableError",
        "create_embedding_service",
    ]


def test_explicit_config_builds_json_safe_cached_synthetic_service(tmp_path: Path) -> None:
    config_path = _write_synthetic_config(tmp_path)

    service = create_embedding_service(config_path)
    status = service.status()
    vectors = service.embed(["alpha", "beta", "alpha"])

    assert isinstance(service, EmbeddingService)
    assert status == {
        "provider": "sentence_transformers",
        "model": "synthetic-test-model",
        "model_key": "sentence_transformers:synthetic-test-model:6",
        "dimension": 6,
        "synthetic": True,
        "cache_enabled": True,
        "cache_db_path": str((config_path.parent / "data/embedding-cache.sqlite3").resolve()),
        "limits": {
            "max_batch_size": 128,
            "max_text_chars": 32_768,
            "max_total_chars": 262_144,
        },
    }
    assert len(vectors) == 3
    assert all(len(vector) == 6 for vector in vectors)
    assert vectors[0] == vectors[2]
    assert (config_path.parent / "data/embedding-cache.sqlite3").is_file()
    json.dumps(status, allow_nan=False)
    json.dumps(vectors, allow_nan=False)


def test_status_returns_an_independent_snapshot(tmp_path: Path) -> None:
    service = create_embedding_service(_write_synthetic_config(tmp_path))

    first = service.status()
    first["limits"]["max_batch_size"] = 1

    assert service.status()["limits"]["max_batch_size"] == 128


def test_omitted_config_uses_rag_config_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_synthetic_config(tmp_path, cache_enabled=False, dimension=4)
    monkeypatch.setenv("RAG_CONFIG_PATH", str(config_path))
    monkeypatch.chdir(tmp_path)

    service = create_embedding_service()

    assert service.status()["dimension"] == 4
    assert service.status()["cache_enabled"] is False
    assert len(service.embed(["configured through env"])[0]) == 4


def test_explicit_config_takes_precedence_over_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_synthetic_config(tmp_path, dimension=5)
    monkeypatch.setenv("RAG_CONFIG_PATH", str(tmp_path / "missing.yaml"))

    service = create_embedding_service(config_path)

    assert service.status()["dimension"] == 5


def test_service_enforces_input_limits_before_backend_call(tmp_path: Path) -> None:
    service = create_embedding_service(
        _write_synthetic_config(tmp_path),
        limits=EmbeddingLimits(max_batch_size=2, max_text_chars=4, max_total_chars=6),
    )

    assert service.embed([]) == []
    with pytest.raises(ValueError, match="sequence of strings"):
        service.embed(cast("list[str]", "text"))
    with pytest.raises(ValueError, match="max_batch_size=2"):
        service.embed(["a", "b", "c"])
    with pytest.raises(ValueError, match=r"texts\[0\] must not be blank"):
        service.embed(["  "])
    with pytest.raises(ValueError, match=r"texts\[1\] must be a string"):
        service.embed(cast("list[str]", ["ok", 3]))
    with pytest.raises(ValueError, match="max_text_chars=4"):
        service.embed(["abcde"])
    with pytest.raises(ValueError, match="max_total_chars=6"):
        service.embed(["abcd", "abc"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_batch_size": 0}, "max_batch_size"),
        ({"max_text_chars": 0}, "max_text_chars"),
        ({"max_text_chars": 5, "max_total_chars": 4}, "max_total_chars"),
    ],
)
def test_embedding_limits_reject_invalid_values(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EmbeddingLimits(**kwargs)


@pytest.mark.parametrize(
    ("texts", "raw_vectors", "message"),
    [
        (["one", "two"], [[1.0, 2.0]], "1 vectors for 2 texts"),
        (["one"], [["bad", 2.0]], "non-numeric vector"),
        (["one"], [[1.0]], "dimension 1"),
        (["one"], [[float("nan"), 2.0]], "non-finite vector"),
    ],
)
def test_service_rejects_malformed_provider_output(
    tmp_path: Path,
    texts: list[str],
    raw_vectors: list[list[object]],
    message: str,
) -> None:
    class MalformedEmbedder:
        dim = 2

        def embed(self, _texts: object) -> list[list[object]]:
            return raw_vectors

    service = _ConfiguredEmbeddingService(
        embedder=cast("object", MalformedEmbedder()),
        settings_obj=Settings(
            synthetic_embeddings=True,
            synthetic_embedding_dim=2,
            data_dir=tmp_path,
            embedding_cache_db_path=str(tmp_path / "cache.sqlite3"),
        ),
        limits=EmbeddingLimits(),
    )

    with pytest.raises(RuntimeError, match=message):
        service.embed(texts)


def test_missing_backend_uses_existing_typed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_synthetic_config(tmp_path)
    payload = config_path.read_text(encoding="utf-8").replace(
        "synthetic_embeddings: true", "synthetic_embeddings: false"
    )
    config_path.write_text(payload, encoding="utf-8")

    def unavailable(_model_name: str, _settings_obj: object) -> object:
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(embedding_factory, "_build_default_st_embedder", unavailable)

    with pytest.raises(
        EmbeddingsBackendUnavailableError,
        match="Dense/hybrid retrieval requires an embeddings backend",
    ):
        create_embedding_service(config_path)


def test_explicit_config_works_in_fresh_process_outside_checkout(tmp_path: Path) -> None:
    config_path = _write_synthetic_config(tmp_path, cache_enabled=False, dimension=3)
    outside = tmp_path / "outside"
    outside.mkdir()
    env = os.environ.copy()
    env.pop("RAG_CONFIG_PATH", None)
    for key in tuple(env):
        if key.startswith("COV_CORE_") or key == "COVERAGE_PROCESS_START":
            env.pop(key)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "from local_rag_backend.integrations.embeddings import create_embedding_service; "
                "service = create_embedding_service(sys.argv[1]); "
                "print(json.dumps({'status': service.status(), "
                "'vectors': service.embed(['outside'])}, allow_nan=False))"
            ),
            str(config_path),
        ],
        cwd=outside,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"]["dimension"] == 3
    assert len(payload["vectors"][0]) == 3

# tests/unit/infrastructure/persistence/vector/test_manifest.py

from pathlib import Path

import pytest

from local_rag_backend.infrastructure.persistence.vector.manifest import (
    build_expected_manifest_config,
    build_manifest,
    expected_manifest_config_from_settings,
    manifest_path_for,
    overwrite_manifest_for_settings,
    purge_index_artifacts,
    read_manifest,
    write_manifest,
)
from local_rag_backend.settings import settings


def test_read_manifest_empty_raises(tmp_path: Path) -> None:
    p = tmp_path / "index_manifest.json"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty manifest"):
        read_manifest(p)


def test_read_manifest_non_object_raises(tmp_path: Path) -> None:
    p = tmp_path / "index_manifest.json"
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        read_manifest(p)


def test_overwrite_manifest_preserves_created_at(tmp_path: Path) -> None:
    idx = tmp_path / "index.faiss"
    mpath = manifest_path_for(idx)

    base = build_expected_manifest_config(
        embedding_backend="openai",
        embedding_model="m1",
        chunker_strategy="chars_v1",
        chunker_version="v1",
    )
    write_manifest(
        mpath,
        build_manifest(
            expected=base,
            dimension=4,
            index_backend="numpy",
            created_at="2026-02-16T00:00:00+00:00",
            updated_at="2026-02-16T00:00:00+00:00",
        ),
    )

    overwrite_manifest_for_settings(
        index_path=idx,
        expected=dict(base, embedding_model="m2"),
        dimension=4,
        index_backend="numpy",
    )

    loaded = read_manifest(mpath)
    assert loaded is not None
    assert loaded["created_at"] == "2026-02-16T00:00:00+00:00"
    assert loaded["embedding_model"] == "m2"


def test_purge_index_artifacts_removes_files(tmp_path: Path) -> None:
    idx = tmp_path / "index.faiss"
    id_map = tmp_path / "id_map.json"
    lock = tmp_path / "index.faiss.lock"
    manifest = manifest_path_for(idx)

    idx.write_bytes(b"x")
    id_map.write_text("[1]", encoding="utf-8")
    lock.write_bytes(b"0")
    write_manifest(
        manifest,
        build_manifest(
            expected=build_expected_manifest_config(
                embedding_backend="openai",
                embedding_model="m1",
                chunker_strategy="chars_v1",
                chunker_version="v1",
            ),
            dimension=4,
            index_backend="numpy",
        ),
    )

    purge_index_artifacts(index_path=idx, id_map_path=id_map)
    assert not idx.exists()
    assert not id_map.exists()
    assert not lock.exists()
    assert not manifest.exists()


def test_expected_manifest_config_from_settings_switches_backend(monkeypatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(settings, "openai_embedding_model", "text-embedding-3-small", raising=False)
    monkeypatch.setattr(settings, "st_embedding_model", "all-MiniLM-L6-v2", raising=False)
    monkeypatch.setattr(settings, "ingest_chunk_strategy", "chars_v1", raising=False)
    monkeypatch.setattr(settings, "ingest_chunker_version", "v3", raising=False)

    cfg_openai = expected_manifest_config_from_settings(settings)
    assert cfg_openai["embedding_backend"] == "openai"
    assert cfg_openai["embedding_model"] == "text-embedding-3-small"
    assert cfg_openai["chunker_strategy"] == "chars_v1"
    assert cfg_openai["chunker_version"] == "v3"

    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    cfg_st = expected_manifest_config_from_settings(settings)
    assert cfg_st["embedding_backend"] == "sentence_transformers"
    assert cfg_st["embedding_model"] == "all-MiniLM-L6-v2"

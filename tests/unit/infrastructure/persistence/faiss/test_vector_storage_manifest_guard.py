# tests/unit/infrastructure/persistence/faiss/test_vector_storage_manifest_guard.py

import pytest

from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
from local_rag_backend.infrastructure.persistence.faiss.index import FaissIndex
from local_rag_backend.infrastructure.persistence.faiss.manifest import (
    manifest_path_for,
    read_manifest,
    write_manifest,
)
from local_rag_backend.settings import settings


def test_upsert_does_not_mutate_index_when_manifest_drifts(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "st_embedding_model", "all-MiniLM-L6-v2", raising=False)

    index_path = tmp_path / "index.faiss"
    id_map_path = tmp_path / "id_map.json"
    vec = FaissVectorStorage(str(index_path), str(id_map_path), dim=4)
    vec.rebuild([], [])

    mpath = manifest_path_for(index_path)
    manifest = read_manifest(mpath)
    assert isinstance(manifest, dict)
    manifest["embedding_model"] = "different-model"
    write_manifest(mpath, manifest)

    with pytest.raises(RuntimeError, match="drift"):
        vec.upsert([1], [[0.0, 0.0, 0.0, 0.0]])

    idx = FaissIndex(index_path, id_map_path, dim=4)
    assert idx.ntotal == 0
    assert idx.id_map == []


def test_delete_does_not_mutate_index_when_manifest_drifts(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "st_embedding_model", "all-MiniLM-L6-v2", raising=False)

    index_path = tmp_path / "index.faiss"
    id_map_path = tmp_path / "id_map.json"
    vec = FaissVectorStorage(str(index_path), str(id_map_path), dim=4)
    vec.rebuild([1], [[0.0, 0.0, 0.0, 0.0]])

    mpath = manifest_path_for(index_path)
    manifest = read_manifest(mpath)
    assert isinstance(manifest, dict)
    manifest["chunker_version"] = "other-version"
    write_manifest(mpath, manifest)

    with pytest.raises(RuntimeError, match="drift"):
        vec.delete([1])

    idx = FaissIndex(index_path, id_map_path, dim=4)
    assert idx.ntotal == 1
    assert idx.id_map == [1]


def test_upsert_fails_closed_when_manifest_is_missing_for_non_empty_legacy_index(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "st_embedding_model", "all-MiniLM-L6-v2", raising=False)

    index_path = tmp_path / "index.faiss"
    id_map_path = tmp_path / "id_map.json"

    # Simulate a legacy pre-manifest index with existing vectors.
    legacy = FaissIndex(index_path, id_map_path, dim=4)
    legacy.rebuild([101], [[0.1, 0.2, 0.3, 0.4]])
    assert manifest_path_for(index_path).exists() is False

    vec = FaissVectorStorage(str(index_path), str(id_map_path), dim=4)
    with pytest.raises(RuntimeError, match="manifest missing for a non-empty index"):
        vec.upsert([102], [[0.5, 0.6, 0.7, 0.8]])

    # Guard: no mutation should happen when we fail closed.
    idx = FaissIndex(index_path, id_map_path, dim=4)
    assert idx.ntotal == 1
    assert idx.id_map == [101]

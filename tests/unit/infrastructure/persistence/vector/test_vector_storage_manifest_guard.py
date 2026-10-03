# tests/unit/infrastructure/persistence/vector/test_vector_storage_manifest_guard.py

import pytest

from local_rag_backend.core.domain.embeddings import EmbeddingIdentity
from local_rag_backend.infrastructure.persistence.vector.index import VectorIndex
from local_rag_backend.infrastructure.persistence.vector.manifest import (
    manifest_path_for,
    read_manifest,
    write_manifest,
)
from local_rag_backend.infrastructure.persistence.vector.storage import VectorStorage
from local_rag_backend.settings import Settings

settings = Settings()


def test_upsert_does_not_mutate_index_when_manifest_drifts(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "st_embedding_model", "all-MiniLM-L6-v2", raising=False)

    index_path = tmp_path / "index.faiss"
    id_map_path = tmp_path / "id_map.json"
    vec = VectorStorage(str(index_path), str(id_map_path), dim=4, settings_obj=settings)
    vec.rebuild([], [])

    mpath = manifest_path_for(index_path)
    manifest = read_manifest(mpath)
    assert isinstance(manifest, dict)
    manifest["embedding_model"] = "different-model"
    write_manifest(mpath, manifest)

    with pytest.raises(RuntimeError, match="drift"):
        vec.upsert(["doc:1"], [[0.0, 0.0, 0.0, 0.0]])

    idx = VectorIndex(index_path, id_map_path, dim=4)
    assert idx.ntotal == 0
    assert idx.id_map == []


def test_delete_does_not_mutate_index_when_manifest_drifts(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "st_embedding_model", "all-MiniLM-L6-v2", raising=False)

    index_path = tmp_path / "index.faiss"
    id_map_path = tmp_path / "id_map.json"
    vec = VectorStorage(str(index_path), str(id_map_path), dim=4, settings_obj=settings)
    vec.rebuild(["doc:1"], [[0.0, 0.0, 0.0, 0.0]])

    mpath = manifest_path_for(index_path)
    manifest = read_manifest(mpath)
    assert isinstance(manifest, dict)
    manifest["chunker_version"] = "other-version"
    write_manifest(mpath, manifest)

    with pytest.raises(RuntimeError, match="drift"):
        vec.delete(["doc:1"])

    idx = VectorIndex(index_path, id_map_path, dim=4)
    assert idx.ntotal == 1
    assert idx.id_map == ["doc:1"]


def test_upsert_fails_closed_when_manifest_is_missing_for_non_empty_unmanaged_index(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "st_embedding_model", "all-MiniLM-L6-v2", raising=False)

    index_path = tmp_path / "index.faiss"
    id_map_path = tmp_path / "id_map.json"

    # Simulate a unmanaged pre-manifest index with existing vectors.
    unmanaged = VectorIndex(index_path, id_map_path, dim=4)
    unmanaged.rebuild(["doc:101"], [[0.1, 0.2, 0.3, 0.4]])
    assert manifest_path_for(index_path).exists() is False

    vec = VectorStorage(str(index_path), str(id_map_path), dim=4, settings_obj=settings)
    with pytest.raises(RuntimeError, match="manifest missing for a non-empty index"):
        vec.upsert(["doc:102"], [[0.5, 0.6, 0.7, 0.8]])

    # Guard: no mutation should happen when we fail closed.
    idx = VectorIndex(index_path, id_map_path, dim=4)
    assert idx.ntotal == 1
    assert idx.id_map == ["doc:101"]


@pytest.mark.parametrize(
    "mutation", ["missing", "version", "synthetic", "implementation", "corrupt"]
)
def test_reads_reject_incompatible_manifests_without_repairing(tmp_path, mutation):
    index_path, id_map_path = tmp_path / "index.npy", tmp_path / "ids.json"
    identity = EmbeddingIdentity("sentence_transformers", "test-model", 2)
    store = VectorStorage(
        str(index_path),
        str(id_map_path),
        dim=2,
        backend="numpy",
        settings_obj=Settings(),
        embedding_identity=identity,
    )
    store.rebuild(["doc"], [[1.0, 0.0]])
    manifest_path = manifest_path_for(index_path)
    manifest = read_manifest(manifest_path)
    if mutation == "missing":
        manifest_path.unlink()
    elif mutation == "corrupt":
        manifest_path.write_text("not json")
    else:
        key, value = {
            "version": ("manifest_version", 1),
            "synthetic": ("synthetic", True),
            "implementation": ("implementation_version", "other"),
        }[mutation]
        manifest[key] = value
        write_manifest(manifest_path, manifest)
    before = manifest_path.read_bytes() if manifest_path.exists() else None
    with pytest.raises(RuntimeError, match="rebuild is required"):
        store.similar([1.0, 0.0], 1)
    assert (manifest_path.read_bytes() if manifest_path.exists() else None) == before

    store.rebuild(["doc"], [[1.0, 0.0]])
    assert store.similar([1.0, 0.0], 1) == [("doc", 1.0)]
    assert read_manifest(manifest_path)["manifest_version"] == 2


def test_empty_index_reads_and_deletes_do_not_create_manifest(tmp_path):
    index_path = tmp_path / "index.npy"
    store = VectorStorage(
        str(index_path),
        str(tmp_path / "ids.json"),
        dim=2,
        backend="numpy",
        settings_obj=Settings(),
    )
    assert store.similar([1.0, 0.0], 1) == []
    assert store.delete(["absent"]) == 0
    store.apply_delta_atomic(delete_ids=["absent"], upserts=[])
    store.upsert([], [])
    assert not manifest_path_for(index_path).exists()

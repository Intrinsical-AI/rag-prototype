"""Live search has no bootstrap, generation, history or provider dependency in sparse mode."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from local_rag_backend.composition import adapters
from local_rag_backend.composition.container import AppContainer
from local_rag_backend.composition.operations import mutate_documents, search
from local_rag_backend.core.domain.retrieval import RetrievalFilter
from local_rag_backend.infrastructure.persistence.sql.base import Base
from local_rag_backend.infrastructure.persistence.sql.document_storage import SqlDocumentStorage
from local_rag_backend.settings import Settings


def _settings(root: Path, mode: str = "sparse") -> Settings:
    return Settings(
        data_dir=root / "data",
        sqlite_url=f"sqlite:///{root / 'documents.sqlite3'}",
        index_path=str(root / "index.npy"),
        id_map_path=str(root / "ids.json"),
        embedding_cache_db_path=str(root / "cache.sqlite3"),
        retrieval_mode=mode,
        vector_backend="numpy",
        synthetic_embeddings=True,
        openai_api_key=None,
        disable_embedding_cache=True,
    )


def _files(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_missing_store_is_refused_without_creating_anything(tmp_path):
    with pytest.raises(FileNotFoundError, match="Private document store missing"):
        search(settings=_settings(tmp_path), question="needle")
    assert _files(tmp_path) == {}


def test_sparse_search_reads_fresh_sql_filters_without_bootstrap_or_providers(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path)
    engine = create_engine(settings.sqlite_url)
    Base.metadata.create_all(engine)
    repository = SqlDocumentStorage(sessionmaker(bind=engine))
    repository.upsert_documents_by_external_id(
        [
            repository.UpsertDoc(
                external_id="one", content="needle alpha", metadata={"scope": "keep"}
            ),
            repository.UpsertDoc(
                external_id="two", content="needle beta", metadata={"scope": "other"}
            ),
        ]
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("search must not bootstrap or construct providers")

    monkeypatch.setattr(adapters, "build_dense_embedder_from_settings", forbidden)
    monkeypatch.setattr(AppContainer, "initialize", forbidden)
    monkeypatch.setattr(AppContainer, "build_generator_from_config", forbidden)
    before = _files(tmp_path)
    result = search(
        settings=settings, question="needle", filters=(RetrievalFilter("scope", ("keep",)),)
    )
    assert [row["document"]["external_id"] for row in result["sources"]] == ["one"]
    assert result["retrieval_mode"] == "sparse"
    assert _files(tmp_path) == before
    repository.upsert_documents_by_external_id(
        [
            repository.UpsertDoc(
                external_id="three", content="novel freshness", metadata={"scope": "keep"}
            ),
        ]
    )
    assert (
        search(settings=settings, question="freshness")["sources"][0]["document"]["external_id"]
        == "three"
    )
    engine.dispose()


@pytest.mark.parametrize("mode", ["dense", "dual", "hybrid"])
def test_vector_modes_search_existing_index_without_writes(tmp_path, mode):
    settings = _settings(tmp_path, mode)
    container = AppContainer.from_settings(settings)
    try:
        mutate_documents(
            container=container,
            source="test",
            payload={
                "op_id": "seed",
                "upserts": [
                    {"external_id": "one", "content": "needle alpha", "scope": "keep"},
                    {"external_id": "two", "content": "needle beta", "scope": "other"},
                ],
            },
        )
        before = _files(tmp_path)
        result = search(
            settings=settings, question="needle", filters=(RetrievalFilter("scope", ("keep",)),)
        )
        assert result["retrieval_mode"] == mode
        assert [row["document"]["external_id"] for row in result["sources"]] == ["one"]
        assert _files(tmp_path) == before
    finally:
        container.close()

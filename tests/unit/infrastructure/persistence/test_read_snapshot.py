from __future__ import annotations

import hashlib
import json
import os
import signal
import sqlite3

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from local_rag_backend.composition import ephemeral
from local_rag_backend.core.domain.retrieval import RetrievalFilter, document_matches_filters
from local_rag_backend.core.domain.types import DocId
from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
    SentenceTransformerEmbedder,
)
from local_rag_backend.infrastructure.persistence.read_snapshot import (
    SnapshotError,
    load_snapshot,
    prepare_snapshot,
)
from local_rag_backend.infrastructure.persistence.shared.mutation_journal import FileMutationJournal
from local_rag_backend.infrastructure.persistence.sql.base import Base
from local_rag_backend.infrastructure.persistence.sql.document_storage import SqlDocumentStorage
from local_rag_backend.infrastructure.persistence.sql.models import Document, QaHistory
from local_rag_backend.infrastructure.persistence.vector.storage import VectorStorage
from local_rag_backend.settings import Settings


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "live.sqlite3"
    settings = Settings(
        sqlite_url=f"sqlite:///{path}",
        data_dir=tmp_path / "live-data",
        retrieval_mode="sparse",
        synthetic_embeddings=True,
        synthetic_embedding_dim=3,
        vector_backend="numpy",
        index_path=str(tmp_path / "live-vector.bin"),
        id_map_path=str(tmp_path / "live-ids.json"),
        ephemeral_snapshot_dir=tmp_path / "frozen",
        ollama_enabled=True,
        openai_api_key=None,
    )
    engine = create_engine(settings.sqlite_url)
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            Document.__table__.insert(),
            [
                {
                    "doc_id": "one",
                    "content": "alpha beta",
                    "external_id": "external-one",
                    "metadata": {"scope": "public"},
                }
            ],
        )
        connection.execute(
            QaHistory.__table__.insert(),
            {"question": "PRIVATE-HISTORY-SECRET-ONLY", "answer": "private answer"},
        )
    engine.dispose()
    return settings


def digests(root):
    return {
        str(path.relative_to(root)): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def test_fresh_snapshot_contains_only_documents_without_history_or_free_pages(source):
    manifest = prepare_snapshot(source, source.ephemeral_snapshot_dir)
    data = (source.ephemeral_snapshot_dir / "documents.sqlite3").read_bytes()
    assert b"PRIVATE-HISTORY-SECRET-ONLY" not in data
    with sqlite3.connect(source.ephemeral_snapshot_dir / "documents.sqlite3") as connection:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall() == [("documents",)]
    assert manifest.documents_count == 1
    assert load_snapshot(source).documents[0].content == "alpha beta"


def test_snapshot_reads_do_not_change_live_or_frozen_files(source, tmp_path):
    prepare_snapshot(source, source.ephemeral_snapshot_dir)
    before = digests(tmp_path)
    snapshot = load_snapshot(source)
    status = ephemeral.snapshot_status(source)
    assert status["health"]["snapshot_ready"] is True
    assert status["health"]["snapshot_id"] == snapshot.manifest.snapshot_id
    assert digests(tmp_path) == before


@pytest.mark.parametrize(
    "metadata", [None, {"kind": "sample"}, {"scope": "stale", "snapshot_id": "stale"}]
)
def test_frozen_scope_filters_match_authoritative_sql_columns(source, metadata):
    engine = create_engine(source.sqlite_url)
    with engine.begin() as connection:
        connection.execute(
            Document.__table__.update().values(
                scope="review-scope", snapshot_id="generation-one", metadata=metadata
            )
        )
    live = SqlDocumentStorage(sessionmaker(bind=engine)).get_all_documents()[0]
    engine.dispose()
    prepare_snapshot(source, source.ephemeral_snapshot_dir)
    frozen = load_snapshot(source).documents[0]
    filters = (
        RetrievalFilter("scope", ("review-scope",)),
        RetrievalFilter("snapshot_id", ("generation-one",)),
    )
    assert document_matches_filters(live, filters)
    assert frozen.metadata == live.metadata
    assert document_matches_filters(frozen, filters)


@pytest.mark.parametrize(
    "damage", ["digest", "sidecar", "unknown", "unsealed", "symlink", "manifest_identity"]
)
def test_unsealed_or_changed_bundle_is_refused(source, damage):
    prepare_snapshot(source, source.ephemeral_snapshot_dir)
    root = source.ephemeral_snapshot_dir
    if damage == "digest":
        with (root / "documents.sqlite3").open("ab") as stream:
            stream.write(b"tampered")
    elif damage == "sidecar":
        (root / "documents.sqlite3-wal").write_bytes(b"private")
    elif damage == "unknown":
        (root / "unknown").write_text("private")
    elif damage == "unsealed":
        (root / "manifest.json").unlink()
    elif damage == "symlink":
        (root / "documents.sqlite3").unlink()
        (root / "documents.sqlite3").symlink_to(source.get_database_path())
    else:
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["documents_count"] = 99
        (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((SnapshotError, OSError)):
        load_snapshot(source)


def test_existing_empty_destination_is_not_overwritten(source):
    source.ephemeral_snapshot_dir.mkdir()
    with pytest.raises(FileExistsError):
        prepare_snapshot(source, source.ephemeral_snapshot_dir)
    assert list(source.ephemeral_snapshot_dir.iterdir()) == []


@pytest.mark.skipif(
    not hasattr(os, "mkfifo") or not hasattr(signal, "setitimer"), reason="POSIX FIFO"
)
@pytest.mark.parametrize("member", ["manifest.json", "documents.sqlite3"])
def test_fifo_member_is_refused_without_waiting_for_a_writer(source, member):
    prepare_snapshot(source, source.ephemeral_snapshot_dir)
    path = source.ephemeral_snapshot_dir / member
    path.unlink()
    os.mkfifo(path)

    def timed_out(signum, frame):
        raise AssertionError("Snapshot reader blocked opening a FIFO")

    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, 1)
    try:
        with pytest.raises(SnapshotError, match="bounded regular file"):
            load_snapshot(source)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize(
    "update",
    [
        {"hybrid_retrieval_alpha": 0.9},
        {"enable_reranker": True},
        {"synthetic_embedding_dim": 16},
        {"vector_backend": "faiss"},
    ],
)
def test_retrieval_settings_drift_refuses_a_sealed_generation(source, update):
    settings = source.model_copy(update={"retrieval_mode": "dual"})
    prepare_snapshot(settings, settings.ephemeral_snapshot_dir)
    with pytest.raises(SnapshotError, match="configuration drift"):
        load_snapshot(settings.model_copy(update=update))


def test_unresolved_journal_prevents_sealing(source, monkeypatch):
    monkeypatch.setattr(FileMutationJournal, "count_incomplete", lambda self: 1)
    with pytest.raises(SnapshotError, match="unresolved mutation"):
        prepare_snapshot(source, source.ephemeral_snapshot_dir)
    assert not (source.ephemeral_snapshot_dir / "manifest.json").exists()


@pytest.mark.parametrize("mode", ["sparse", "dense", "dual", "hybrid"])
def test_ephemeral_modes_reuse_retrieval_without_history_or_storage_writes(
    source, tmp_path, mode, monkeypatch
):
    settings = source.model_copy(update={"retrieval_mode": mode})
    if mode in {"dense", "hybrid"}:
        embedder = SentenceTransformerEmbedder(settings_obj=settings)
        storage = VectorStorage(
            settings.index_path,
            settings.id_map_path,
            dim=embedder.dim,
            backend="numpy",
            settings_obj=settings,
            embedding_identity=embedder.identity,
        )
        storage.rebuild([DocId("one")], embedder.embed(["alpha beta"]))
    prepare_snapshot(settings, settings.ephemeral_snapshot_dir)

    class Generator:
        def generate(self, question, contexts):
            assert contexts == ["alpha beta"]
            return "synthetic answer"

    def generator_factory(**kwargs):
        assert kwargs["openai_generator_factory"].keywords["max_retries"] == 0
        return Generator()

    monkeypatch.setattr(ephemeral, "build_generator_from_settings", generator_factory)
    before = digests(tmp_path)
    with ephemeral.build_ephemeral_service(settings) as (service, snapshot_id):
        result = service.ask("alpha", retrieval_mode=mode, persist_history=False)
    assert result["answer"] == "synthetic answer"
    assert result["docs"][0].id == "one"
    assert snapshot_id == load_snapshot(settings).manifest.snapshot_id
    assert digests(tmp_path) == before


def test_status_missing_store_does_not_create_a_schema(source, tmp_path):
    source.get_database_path().unlink()
    before = digests(tmp_path)
    result = ephemeral.snapshot_status(source)
    assert result["runtime"]["live_store"] == "missing"
    assert result["health"]["snapshot_ready"] is False
    assert digests(tmp_path) == before


def test_empty_faiss_snapshot_matches_live_storage_and_validates_queries(source):
    pytest.importorskip("faiss")
    from local_rag_backend.infrastructure.persistence.vector.read_snapshot import (
        SnapshotVectorStorage,
    )

    settings = source.model_copy(update={"retrieval_mode": "dense", "vector_backend": "faiss"})
    engine = create_engine(settings.sqlite_url)
    with engine.begin() as connection:
        connection.execute(Document.__table__.delete())
    engine.dispose()
    embedder = SentenceTransformerEmbedder(settings_obj=settings)
    live = VectorStorage(
        settings.index_path,
        settings.id_map_path,
        dim=embedder.dim,
        backend="faiss",
        settings_obj=settings,
        embedding_identity=embedder.identity,
    )
    live.rebuild([], [])
    prepare_snapshot(settings, settings.ephemeral_snapshot_dir)
    frozen = SnapshotVectorStorage(load_snapshot(settings))
    query = [0.0] * embedder.dim
    assert frozen.ntotal == 0
    assert live.similar(query, 3) == []
    assert frozen.similar(query, 3) == []
    with pytest.raises(SnapshotError, match="dimension or values"):
        frozen.similar([float("nan")] * embedder.dim, 3)

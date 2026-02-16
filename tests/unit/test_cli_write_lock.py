from __future__ import annotations

from contextlib import contextmanager

from click.testing import CliRunner

from local_rag_backend.cli import cli
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage
from local_rag_backend.settings import settings


def test_cli_upsert_docs_runs_under_multi_store_lock(in_memory_sqlite, monkeypatch):
    lock_entries = 0

    @contextmanager
    def _fake_lock():
        nonlocal lock_entries
        lock_entries += 1
        yield

    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(
        "local_rag_backend.core.services.write_lock.multi_store_write_lock",
        _fake_lock,
        raising=True,
    )

    result = CliRunner().invoke(
        cli, ["upsert-docs", "--external-id", "doc-1", "--content", "hello"]
    )
    assert result.exit_code == 0, result.output
    assert lock_entries == 1


def test_cli_delete_docs_runs_under_multi_store_lock(in_memory_sqlite, monkeypatch):
    lock_entries = 0

    @contextmanager
    def _fake_lock():
        nonlocal lock_entries
        lock_entries += 1
        yield

    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(
        "local_rag_backend.core.services.write_lock.multi_store_write_lock",
        _fake_lock,
        raising=True,
    )

    doc_id = SqlDocumentStorage().store_documents(["to-delete"])[0]
    result = CliRunner().invoke(cli, ["delete-docs", str(doc_id)])
    assert result.exit_code == 0, result.output
    assert lock_entries == 1


def test_cli_ingest_runs_mutation_batches_under_multi_store_lock(
    in_memory_sqlite, tmp_path, monkeypatch
):
    lock_entries = 0

    @contextmanager
    def _fake_lock():
        nonlocal lock_entries
        lock_entries += 1
        yield

    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(
        "local_rag_backend.core.services.write_lock.multi_store_write_lock",
        _fake_lock,
        raising=True,
    )

    root = tmp_path / "in"
    root.mkdir()
    (root / "a.txt").write_text("hello world", encoding="utf-8")
    (root / "b.txt").write_text("hello world 2", encoding="utf-8")

    result = CliRunner().invoke(cli, ["ingest", str(root), "--no-magic"])
    assert result.exit_code == 0, result.output
    # Ingest now batches file mutations, so multiple files can share one lock window.
    assert lock_entries == 1


def test_cli_delete_docs_dense_does_not_require_embedder_when_index_delete_succeeds(
    in_memory_sqlite, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)

    embedder_calls = 0

    def _boom_embedder(*_args, **_kwargs):
        nonlocal embedder_calls
        embedder_calls += 1
        raise RuntimeError("embedder should not be called on successful delete path")

    class DummyVec:
        def __init__(self, *_args, **_kwargs) -> None:
            return None

        def delete(self, ids):
            return len(list(ids))

    monkeypatch.setattr(
        "local_rag_backend.infrastructure.embeddings.sentence_transformers.SentenceTransformerEmbedder",
        _boom_embedder,
        raising=True,
    )
    monkeypatch.setattr(
        "local_rag_backend.infrastructure.persistence.faiss.faiss_.FaissVectorStorage",
        lambda *_a, **_k: DummyVec(),
        raising=True,
    )

    doc_id = SqlDocumentStorage().store_documents(["to-delete-dense"])[0]
    result = CliRunner().invoke(cli, ["delete-docs", str(doc_id)])
    assert result.exit_code == 0, result.output
    assert embedder_calls == 0

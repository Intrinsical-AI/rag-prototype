from __future__ import annotations

import json
from contextlib import contextmanager

from click.testing import CliRunner

from local_rag_backend.cli import cli
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.settings import settings


def test_cli_mutate_docs_runs_under_multi_store_lock(in_memory_sqlite, tmp_path, monkeypatch):
    lock_entries = 0

    @contextmanager
    def _fake_lock():
        nonlocal lock_entries
        lock_entries += 1
        yield

    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(
        "local_rag_backend.infrastructure.concurrency.locks.write_lock.multi_store_write_lock",
        _fake_lock,
        raising=True,
    )

    payload = tmp_path / "mutate_lock.json"
    payload.write_text(
        json.dumps({"upserts": [{"external_id": "doc-1", "content": "hello"}]}),
        encoding="utf-8",
    )
    result = CliRunner().invoke(cli, ["mutate-docs", "--json", str(payload)])
    assert result.exit_code == 0, result.output
    assert lock_entries == 0


def test_cli_mutate_delete_ids_runs_under_multi_store_lock(in_memory_sqlite, tmp_path, monkeypatch):
    lock_entries = 0

    @contextmanager
    def _fake_lock():
        nonlocal lock_entries
        lock_entries += 1
        yield

    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(
        "local_rag_backend.infrastructure.concurrency.locks.write_lock.multi_store_write_lock",
        _fake_lock,
        raising=True,
    )

    doc_id = SqlDocumentStorage().store_documents(["to-delete"])[0]
    payload = tmp_path / "mutate_delete_ids_lock.json"
    payload.write_text(json.dumps({"delete_ids": [str(doc_id)]}), encoding="utf-8")

    result = CliRunner().invoke(cli, ["mutate-docs", "--json", str(payload)])
    assert result.exit_code == 0, result.output
    assert lock_entries == 0


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
        "local_rag_backend.infrastructure.concurrency.locks.write_lock.multi_store_write_lock",
        _fake_lock,
        raising=True,
    )

    root = tmp_path / "in"
    root.mkdir()
    (root / "a.txt").write_text("hello world", encoding="utf-8")
    (root / "b.txt").write_text("hello world 2", encoding="utf-8")

    result = CliRunner().invoke(cli, ["ingest", str(root), "--no-magic"])
    assert result.exit_code == 0, result.output
    assert lock_entries == 0


def test_cli_mutate_delete_ids_dense_does_not_require_embedder_when_no_upserts(
    in_memory_sqlite, tmp_path, monkeypatch
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

        def apply_delta_atomic(self, *, delete_ids, upserts):
            assert list(upserts) == []
            return None

    monkeypatch.setattr(
        "local_rag_backend.composition.factory.SentenceTransformerEmbedder",
        _boom_embedder,
        raising=True,
    )
    monkeypatch.setattr(
        "local_rag_backend.composition.factory.VectorStorage",
        lambda *_a, **_k: DummyVec(),
        raising=True,
    )

    doc_id = SqlDocumentStorage().store_documents(["to-delete-dense"])[0]
    payload = tmp_path / "mutate_delete_dense_ids.json"
    payload.write_text(json.dumps({"delete_ids": [str(doc_id)]}), encoding="utf-8")
    result = CliRunner().invoke(cli, ["mutate-docs", "--json", str(payload)])
    assert result.exit_code == 0, result.output
    assert embedder_calls == 0


def test_cli_mutate_delete_external_ids_dense_does_not_require_embedder_when_no_upserts(
    in_memory_sqlite, tmp_path, monkeypatch
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

        def apply_delta_atomic(self, *, delete_ids, upserts):
            assert list(upserts) == []
            return None

    monkeypatch.setattr(
        "local_rag_backend.composition.factory.SentenceTransformerEmbedder",
        _boom_embedder,
        raising=True,
    )
    monkeypatch.setattr(
        "local_rag_backend.composition.factory.VectorStorage",
        lambda *_a, **_k: DummyVec(),
        raising=True,
    )

    repo = SqlDocumentStorage()
    repo.upsert_documents_by_external_id(
        [SqlDocumentStorage.UpsertDoc(external_id="doc-ext-ok", content="to-delete-dense")]
    )

    payload = tmp_path / "mutate_delete_dense_external_ids.json"
    payload.write_text(json.dumps({"delete_external_ids": ["doc-ext-ok"]}), encoding="utf-8")
    result = CliRunner().invoke(cli, ["mutate-docs", "--json", str(payload)])
    assert result.exit_code == 0, result.output
    assert embedder_calls == 0

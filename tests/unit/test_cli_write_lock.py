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


def test_cli_ingest_runs_each_file_mutation_under_multi_store_lock(
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

    result = CliRunner().invoke(cli, ["ingest", str(root), "--no-magic"])
    assert result.exit_code == 0, result.output
    assert lock_entries == 1

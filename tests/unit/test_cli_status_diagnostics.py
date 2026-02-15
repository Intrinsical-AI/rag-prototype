# tests/unit/test_cli_status_diagnostics.py

import re

from click.testing import CliRunner

from local_rag_backend.cli import cli
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage
from local_rag_backend.settings import settings


def test_rag_status_reports_document_count(in_memory_sqlite, monkeypatch):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    SqlDocumentStorage().store_documents(["a", "b"])

    r = CliRunner().invoke(cli, ["status"])
    assert r.exit_code == 0, r.output
    assert re.search(r"Documents:\s+2\b", r.output)


def test_rag_status_reports_missing_index_in_dense_mode(in_memory_sqlite, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "index_path", str(tmp_path / "missing.faiss"), raising=False)
    monkeypatch.setattr(settings, "id_map_path", str(tmp_path / "missing.json"), raising=False)

    r = CliRunner().invoke(cli, ["status"])
    assert r.exit_code == 0, r.output
    assert "Index:" in r.output
    assert "missing" in r.output

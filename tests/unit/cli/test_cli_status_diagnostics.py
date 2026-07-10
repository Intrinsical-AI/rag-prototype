# tests/unit/cli/test_cli_status_diagnostics.py

import re
from types import SimpleNamespace

from click.testing import CliRunner

from local_rag_backend.cli import cli
from local_rag_backend.cli_commands import index as index_cmd_module
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
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


def test_rag_status_reports_manifest_missing_in_dense_mode(in_memory_sqlite, tmp_path, monkeypatch):
    from local_rag_backend.infrastructure.persistence.vector.index import VectorIndex

    monkeypatch.setattr(settings, "openai_api_key", "DUMMY", raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "index_path", str(tmp_path / "index.faiss"), raising=False)
    monkeypatch.setattr(settings, "id_map_path", str(tmp_path / "id_map.json"), raising=False)

    # Create index + id_map but no manifest.
    VectorIndex(settings.index_path, settings.id_map_path, dim=4).rebuild([], [])

    r = CliRunner().invoke(cli, ["status"])
    assert r.exit_code == 0, r.output
    assert "Index:" in r.output
    assert "drift" in r.output


def test_status_keeps_exit_code_zero_when_diagnostics_counts_fail(monkeypatch) -> None:
    class _Diagnostics:
        def get_documents_count(self):
            raise RuntimeError("documents backend offline")

        def get_history_count(self):
            raise RuntimeError("history backend offline")

    class _Container:
        settings_obj = settings

        def build_health_readiness_bundle(self):
            return SimpleNamespace(diagnostics=_Diagnostics(), expected_manifest=None)

    monkeypatch.setattr(
        index_cmd_module, "ensure_sqlite_schema_for_cli", lambda: None, raising=True
    )
    monkeypatch.setattr(index_cmd_module, "get_cli_container", lambda: _Container(), raising=True)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    result = CliRunner().invoke(index_cmd_module.status_cmd)

    assert result.exit_code == 0, result.output
    assert "[ERROR] documents backend offline" in result.output
    assert "[WARN] history backend offline" in result.output


def test_status_keeps_exit_code_zero_when_retrieval_stats_fail(monkeypatch) -> None:
    class _Diagnostics:
        def get_documents_count(self):
            return 3

        def get_history_count(self):
            return 2

        def get_retrieval_index_stats(self, **kwargs):
            _ = kwargs
            raise RuntimeError("remote backend offline")

    class _Container:
        settings_obj = settings

        def build_health_readiness_bundle(self):
            return SimpleNamespace(diagnostics=_Diagnostics(), expected_manifest=None)

    monkeypatch.setattr(
        index_cmd_module, "ensure_sqlite_schema_for_cli", lambda: None, raising=True
    )
    monkeypatch.setattr(index_cmd_module, "get_cli_container", lambda: _Container(), raising=True)
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)

    result = CliRunner().invoke(index_cmd_module.status_cmd)

    assert result.exit_code == 0, result.output
    assert "Documents:" in result.output
    assert "[ERROR] remote backend offline" in result.output

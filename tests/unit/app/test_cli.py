# tests/unit/app/test_cli.py

from click.testing import CliRunner

from local_rag_backend import cli as cli_module
from local_rag_backend.settings import settings


def test_cli_status(monkeypatch, tmp_path):
    # Prepare fake files
    db_file = tmp_path / "app.db"
    db_file.write_text("", encoding="utf-8")
    index_file = tmp_path / "index.faiss"
    index_file.write_text("", encoding="utf-8")
    csv_file = tmp_path / "faq.csv"
    csv_file.write_text("Q;A\nq;a\n", encoding="utf-8")

    # Point settings to our temp files
    monkeypatch.setattr(settings, "app_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(settings, "app_port", 9999, raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "debug", False, raising=False)
    monkeypatch.setattr(settings, "index_path", str(index_file), raising=False)
    monkeypatch.setattr(settings, "id_map_path", str(tmp_path / "id.json"), raising=False)
    monkeypatch.setattr(settings, "faq_csv", str(csv_file), raising=False)

    # Point sqlite_url so get_database_path() resolves to our tmp db file
    monkeypatch.setattr(settings, "sqlite_url", f"sqlite:///{db_file}", raising=False)

    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["status"])  # click group
    assert result.exit_code == 0
    out = result.output
    # Basic sanity checks
    assert "Intrinsical RAG Prototype - System Status" in out
    assert "127.0.0.1:9999" in out
    assert "Retrieval Mode: sparse" in out
    assert "Database:" in out and "YES" in out
    assert "FAISS index:" in out and "YES" in out
    assert "Sample data:" in out and "YES" in out

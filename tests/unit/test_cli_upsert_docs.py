# tests/unit/test_cli_upsert_docs.py

import json

from click.testing import CliRunner

from local_rag_backend.cli import cli
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage
from local_rag_backend.settings import settings


def test_cli_upsert_docs_from_json_file(in_memory_sqlite, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    p = tmp_path / "docs.json"
    p.write_text(
        json.dumps(
            [{"external_id": "doc-1", "content": "hello"}, {"external_id": "doc-2", "content": "x"}]
        ),
        encoding="utf-8",
    )

    r = CliRunner().invoke(cli, ["upsert-docs", "--json", str(p)])
    assert r.exit_code == 0, r.output
    assert "inserted=2" in r.output

    docs = SqlDocumentStorage().get_all_documents()
    assert len(docs) == 2
    assert {d.external_id for d in docs} == {"doc-1", "doc-2"}

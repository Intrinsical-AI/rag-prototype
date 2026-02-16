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


def test_cli_upsert_docs_dense_embed_failure_does_not_persist_sql(in_memory_sqlite, monkeypatch):
    class BadEmbedder:
        dim = 4

        def embed(self, texts):
            raise RuntimeError("embed fail")

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(
        "local_rag_backend.infrastructure.embeddings.openai.OpenAIEmbedder",
        lambda *a, **k: BadEmbedder(),
        raising=True,
    )

    r = CliRunner().invoke(
        cli,
        ["upsert-docs", "--external-id", "doc-1", "--content", "hello"],
    )
    assert r.exit_code == 1
    assert "embed fail" in r.output
    assert SqlDocumentStorage().get_all_documents() == []


def test_cli_upsert_docs_failure_still_invalidates_cached_rag_service(
    in_memory_sqlite, monkeypatch
):
    class BadEmbedder:
        dim = 4

        def embed(self, texts):
            raise RuntimeError("embed fail")

    reset_calls = 0

    def _count_reset() -> None:
        nonlocal reset_calls
        reset_calls += 1

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(
        "local_rag_backend.infrastructure.embeddings.openai.OpenAIEmbedder",
        lambda *a, **k: BadEmbedder(),
        raising=True,
    )
    monkeypatch.setattr(
        "local_rag_backend.cli._reset_rag_service_best_effort", _count_reset, raising=True
    )

    r = CliRunner().invoke(
        cli,
        ["upsert-docs", "--external-id", "doc-1", "--content", "hello"],
    )
    assert r.exit_code == 1
    assert "embed fail" in r.output
    assert reset_calls == 1

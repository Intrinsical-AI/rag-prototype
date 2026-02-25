import json

from click.testing import CliRunner

from local_rag_backend.cli import cli
from local_rag_backend.infrastructure.persistence.sql.alchemy_engine import SqlDocumentStorage
from local_rag_backend.settings import settings


def test_cli_mutate_docs_from_json_file(in_memory_sqlite, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    p = tmp_path / "mutate.json"
    p.write_text(
        json.dumps(
            {
                "upserts": [
                    {"external_id": "doc-1", "content": "hello"},
                    {"external_id": "doc-2", "content": "x"},
                ]
            }
        ),
        encoding="utf-8",
    )

    r = CliRunner().invoke(cli, ["mutate-docs", "--json", str(p)])
    assert r.exit_code == 0, r.output
    assert "inserted=2" in r.output

    docs = SqlDocumentStorage().get_all_documents()
    assert len(docs) == 2
    assert {d.external_id for d in docs} == {"doc-1", "doc-2"}


def test_cli_mutate_docs_dense_embed_failure_does_not_persist_sql(
    in_memory_sqlite, tmp_path, monkeypatch
):
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

    p = tmp_path / "mutate_dense_fail.json"
    p.write_text(
        json.dumps({"upserts": [{"external_id": "doc-1", "content": "hello"}]}),
        encoding="utf-8",
    )
    r = CliRunner().invoke(cli, ["mutate-docs", "--json", str(p)])
    assert r.exit_code == 1
    assert "embed fail" in r.output
    assert SqlDocumentStorage().get_all_documents() == []


def test_cli_mutate_docs_failure_still_invalidates_cached_rag_service(
    in_memory_sqlite, tmp_path, monkeypatch
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
        "local_rag_backend.cli_commands.runtime._reset_rag_service_best_effort",
        _count_reset,
        raising=True,
    )

    p = tmp_path / "mutate_dense_reset.json"
    p.write_text(
        json.dumps({"upserts": [{"external_id": "doc-1", "content": "hello"}]}),
        encoding="utf-8",
    )
    r = CliRunner().invoke(cli, ["mutate-docs", "--json", str(p)])
    assert r.exit_code == 1
    assert "embed fail" in r.output
    assert reset_calls == 1


def test_cli_mutate_docs_rejects_duplicate_external_id_before_embedding(
    in_memory_sqlite, tmp_path, monkeypatch
):
    class CountingEmbedder:
        dim = 4

        def __init__(self):
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            return [[0.0, 0.0, 0.0, 0.0] for _ in texts]

    embedder = CountingEmbedder()
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(
        "local_rag_backend.infrastructure.embeddings.openai.OpenAIEmbedder",
        lambda *a, **k: embedder,
        raising=True,
    )

    p = tmp_path / "dup_docs.json"
    p.write_text(
        json.dumps(
            {
                "upserts": [
                    {"external_id": "doc-1", "content": "hello"},
                    {"external_id": "doc-1", "content": "world"},
                ]
            }
        ),
        encoding="utf-8",
    )

    r = CliRunner().invoke(cli, ["mutate-docs", "--json", str(p)])
    assert r.exit_code == 1
    assert "external_id values in upserts must be unique" in r.output
    assert embedder.calls == 0
    assert SqlDocumentStorage().get_all_documents() == []

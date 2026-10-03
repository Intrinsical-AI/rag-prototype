import json

from click.testing import CliRunner
from support.container import override_container

from local_rag_backend.cli import cli
from local_rag_backend.composition import factory
from local_rag_backend.core.domain.embeddings import EmbeddingIdentity
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.settings import get_settings

settings = get_settings()


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

    docs = SqlDocumentStorage(session_factory=in_memory_sqlite).get_all_documents()
    assert len(docs) == 2
    assert {d.external_id for d in docs} == {"doc-1", "doc-2"}


def test_cli_mutate_rejects_blank_upsert_before_deleting(in_memory_sqlite, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    repo = SqlDocumentStorage(session_factory=in_memory_sqlite)
    repo.upsert_documents_by_external_id(
        [repo.UpsertDoc(external_id="keep", content="keep this document")]
    )
    payload = tmp_path / "invalid-mutation.json"
    payload.write_text(
        json.dumps(
            {
                "upserts": [{"external_id": "new", "content": " "}],
                "delete_external_ids": ["keep"],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli, ["mutate-docs", "--json", str(payload)])

    assert result.exit_code == 1
    assert "content must not be blank" in result.output
    assert {doc.external_id for doc in repo.get_all_documents()} == {"keep"}


def test_cli_mutate_docs_dense_embed_failure_does_not_persist_sql(
    in_memory_sqlite, tmp_path, monkeypatch
):
    class BadEmbedder:
        identity = EmbeddingIdentity(provider="openai", model="text-embedding-3-small", dimension=4)
        dim = 4

        def embed(self, texts):
            raise RuntimeError("embed fail")

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    override_container(monkeypatch, openai_embedder_factory=lambda *a, **k: BadEmbedder())

    p = tmp_path / "mutate_dense_fail.json"
    p.write_text(
        json.dumps({"upserts": [{"external_id": "doc-1", "content": "hello"}]}),
        encoding="utf-8",
    )
    r = CliRunner().invoke(cli, ["mutate-docs", "--json", str(p)])
    assert r.exit_code == 1
    assert "embed fail" in r.output
    assert SqlDocumentStorage(session_factory=in_memory_sqlite).get_all_documents() == []


def test_cli_mutate_docs_failure_still_invalidates_cached_rag_service(
    in_memory_sqlite, tmp_path, monkeypatch
):
    class BadEmbedder:
        identity = EmbeddingIdentity(provider="openai", model="text-embedding-3-small", dimension=4)
        dim = 4

        def embed(self, texts):
            raise RuntimeError("embed fail")

    reset_calls = 0

    def _count_reset() -> None:
        nonlocal reset_calls
        reset_calls += 1

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    override_container(monkeypatch, openai_embedder_factory=lambda *a, **k: BadEmbedder())
    monkeypatch.setattr(
        factory.get_app_context().container,
        "clear_local_rag_service_cache",
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
    assert reset_calls >= 1


def test_cli_mutate_docs_rejects_duplicate_external_id_before_embedding(
    in_memory_sqlite, tmp_path, monkeypatch
):
    class CountingEmbedder:
        identity = EmbeddingIdentity(provider="openai", model="text-embedding-3-small", dimension=4)
        dim = 4

        def __init__(self):
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            return [[0.0, 0.0, 0.0, 0.0] for _ in texts]

    embedder = CountingEmbedder()
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    override_container(monkeypatch, openai_embedder_factory=lambda *a, **k: embedder)

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
    assert SqlDocumentStorage(session_factory=in_memory_sqlite).get_all_documents() == []

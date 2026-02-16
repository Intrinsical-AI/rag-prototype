# tests/unit/test_cli_ingest.py

from __future__ import annotations

from click.testing import CliRunner

from local_rag_backend.cli import cli
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage
from local_rag_backend.settings import settings


def test_cli_ingest_dir_mixed_is_idempotent_and_deletes_stale_chunks(
    in_memory_sqlite, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    # Make chunking deterministic and small so we can assert chunk counts easily.
    monkeypatch.setattr(settings, "ingest_chunk_chars", 10, raising=False)
    monkeypatch.setattr(settings, "ingest_chunk_overlap", 0, raising=False)
    monkeypatch.setattr(settings, "csv_has_header", True, raising=False)

    root = tmp_path / "in"
    root.mkdir()
    (root / "a.txt").write_text(
        "0123456789ABCDEFGHIJXXXXX", encoding="utf-8"
    )  # 25 chars -> 3 chunks
    (root / "b.md").write_text("# T\n\nHi", encoding="utf-8")  # small -> 1 chunk
    (root / "c.csv").write_text("title;body\nT1;B1\nT2;B2\n", encoding="utf-8")  # 2 rows -> 2 docs
    (root / "bin.dat").write_bytes(b"\x00\x01\x02\x03")  # binary -> skipped

    r1 = CliRunner().invoke(cli, ["ingest", str(root), "--no-magic"])
    assert r1.exit_code == 0, r1.output

    docs1 = SqlDocumentStorage().get_all_documents()
    assert len(docs1) == 6  # 3 (txt) + 1 (md) + 2 (csv rows)
    assert all(d.external_id and d.external_id.startswith("file:") for d in docs1)
    assert sum(1 for d in docs1 if (d.source_id or "").endswith("a.txt")) == 3
    # Prepared metadata for future parent-doc retrieval.
    a_chunks = [d for d in docs1 if (d.source_id or "").endswith("a.txt")]
    assert all((d.metadata or {}).get("chunk_index") is not None for d in a_chunks)
    assert all((d.metadata or {}).get("parent_doc_id") is not None for d in a_chunks)
    assert all((d.metadata or {}).get("chunk_start_char") is not None for d in a_chunks)
    assert all((d.metadata or {}).get("chunk_end_char") is not None for d in a_chunks)

    r2 = CliRunner().invoke(cli, ["ingest", str(root), "--no-magic"])
    assert r2.exit_code == 0, r2.output
    assert "inserted=0" in r2.output

    docs2 = SqlDocumentStorage().get_all_documents()
    assert len(docs2) == 6  # idempotent

    # Shrink file so it now produces only 1 chunk; old chunks should be deleted.
    (root / "a.txt").write_text("short", encoding="utf-8")
    r3 = CliRunner().invoke(cli, ["ingest", str(root), "--no-magic"])
    assert r3.exit_code == 0, r3.output

    docs3 = SqlDocumentStorage().get_all_documents()
    assert len(docs3) == 4  # 1 (txt) + 1 (md) + 2 (csv)
    assert sum(1 for d in docs3 if (d.source_id or "").endswith("a.txt")) == 1


def test_cli_ingest_dense_embed_failure_does_not_persist_sql(
    in_memory_sqlite, tmp_path, monkeypatch
):
    class BadEmbedder:
        dim = 4

        def embed(self, texts):
            raise RuntimeError("embed fail")

    class DummyVec:
        def __init__(self, *a, **k):
            return None

        def delete(self, ids):
            return None

        def upsert(self, ids, vectors):
            return None

        def rebuild(self, ids, vectors):
            return None

    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(
        "local_rag_backend.infrastructure.embeddings.openai.OpenAIEmbedder",
        lambda *a, **k: BadEmbedder(),
        raising=True,
    )
    monkeypatch.setattr(
        "local_rag_backend.infrastructure.persistence.faiss.faiss_.FaissVectorStorage",
        lambda *a, **k: DummyVec(),
        raising=True,
    )

    root = tmp_path / "in"
    root.mkdir()
    (root / "a.txt").write_text("hello", encoding="utf-8")

    r = CliRunner().invoke(cli, ["ingest", str(root), "--no-magic"])
    assert r.exit_code == 1
    assert "embed fail" in r.output
    assert SqlDocumentStorage().get_all_documents() == []


def test_cli_ingest_accepts_utf8_non_ascii_text(in_memory_sqlite, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "ingest_chunk_chars", 10_000, raising=False)
    monkeypatch.setattr(settings, "ingest_chunk_overlap", 0, raising=False)

    root = tmp_path / "in"
    root.mkdir()
    text_file = root / "es.txt"
    text_file.write_text("¿Cómo está? Información útil para RAG.", encoding="utf-8")

    r = CliRunner().invoke(cli, ["ingest", str(root), "--no-magic"])
    assert r.exit_code == 0, r.output

    docs = SqlDocumentStorage().get_all_documents()
    assert len(docs) == 1
    assert (docs[0].source_id or "").endswith("es.txt")

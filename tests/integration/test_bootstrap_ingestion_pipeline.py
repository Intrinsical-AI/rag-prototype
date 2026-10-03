import csv

import pytest

from local_rag_backend.core.domain.embeddings import EmbeddingIdentity
from local_rag_backend.scripts.sample_data_ingestion import run_sample_data_ingestion
from local_rag_backend.settings import get_settings

settings = get_settings()


def test_bootstrap_with_canonical_mutation_sparse_mode(tmp_path, monkeypatch, caplog):
    """Bootstrap should ingest CSV through canonical durable mutation flow (sparse)."""
    # Create test CSV
    csv_file = tmp_path / "faq.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["Q", "A"])
        writer.writerow(["¿Qué es RAG?", "Es Retrieval-Augmented Generation."])
        writer.writerow(["¿Cómo funciona?", "Combina búsqueda y generación."])

    # Configure settings for sparse mode
    monkeypatch.setattr(settings, "faq_csv", str(csv_file), raising=False)
    monkeypatch.setattr(settings, "csv_has_header", True, raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "sqlite_url", f"sqlite:///{tmp_path}/app.db", raising=False)
    monkeypatch.setattr(settings, "ingest_chunk_chars", 1200, raising=False)
    monkeypatch.setattr(settings, "ingest_chunk_overlap", 200, raising=False)

    import logging

    with caplog.at_level(logging.INFO):
        run_sample_data_ingestion(settings_obj=settings)

    assert any("Ingested" in m for m in caplog.messages)
    assert any("into SQLite" in m for m in caplog.messages)

    # Verify documents were stored
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
    from local_rag_backend.infrastructure.persistence.sql.base import Base

    engine = create_engine(settings.sqlite_url, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    doc_repo = SqlDocumentStorage(session_factory=SessionLocal)
    docs = doc_repo.get_all_documents()

    assert len(docs) == 2
    assert any("RAG" in doc.content for doc in docs)
    assert any("funciona" in doc.content for doc in docs)
    assert {doc.content for doc in docs} == {
        "¿Qué es RAG?\n\nEs Retrieval-Augmented Generation.",
        "¿Cómo funciona?\n\nCombina búsqueda y generación.",
    }


def test_bootstrap_with_canonical_mutation_dense_mode(tmp_path, monkeypatch, caplog):
    """Bootstrap should ingest CSV through canonical durable mutation flow (dense)."""

    class DummyEmbedder:
        dim = 4
        identity = EmbeddingIdentity(
            provider="sentence_transformers", model="all-MiniLM-L6-v2", dimension=4
        )

        def embed(self, texts):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    # Create test CSV
    csv_file = tmp_path / "faq.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["Q", "A"])
        writer.writerow(["Test Question", "Test Answer"])

    # Configure settings for dense mode
    monkeypatch.setattr(settings, "faq_csv", str(csv_file), raising=False)
    monkeypatch.setattr(settings, "csv_has_header", True, raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "sqlite_url", f"sqlite:///{tmp_path}/app.db", raising=False)
    monkeypatch.setattr(settings, "index_path", str(tmp_path / "idx.faiss"), raising=False)
    monkeypatch.setattr(settings, "id_map_path", str(tmp_path / "id.json"), raising=False)
    monkeypatch.setattr(settings, "ingest_chunk_chars", 50, raising=False)
    monkeypatch.setattr(settings, "ingest_chunk_overlap", 10, raising=False)

    # Mock embedder and FAISS
    monkeypatch.setattr(
        "local_rag_backend.composition.container.SentenceTransformerEmbedder",
        lambda model_name=None, settings_obj=None: DummyEmbedder(),
        raising=True,
    )
    monkeypatch.setattr(settings, "vector_backend", "numpy")

    import logging

    with caplog.at_level(logging.INFO):
        run_sample_data_ingestion(settings_obj=settings)

    assert any("Ingested" in m for m in caplog.messages)
    assert any("into SQLite" in m for m in caplog.messages)


def test_bootstrap_with_custom_chunking_settings(tmp_path, monkeypatch, capsys):
    """Test that custom chunking settings are applied."""
    # Create CSV with long content
    csv_file = tmp_path / "faq.csv"
    long_content = "A" * 200  # Long content that will be chunked
    with csv_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["Q", "A"])
        writer.writerow(["Long Question", long_content])

    # Configure small chunk size to force chunking
    monkeypatch.setattr(settings, "faq_csv", str(csv_file), raising=False)
    monkeypatch.setattr(settings, "csv_has_header", True, raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "sqlite_url", f"sqlite:///{tmp_path}/app.db", raising=False)
    monkeypatch.setattr(settings, "ingest_chunk_chars", 50, raising=False)  # Small chunks
    monkeypatch.setattr(settings, "ingest_chunk_overlap", 10, raising=False)

    run_sample_data_ingestion(settings_obj=settings)

    # Verify multiple chunks were created
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
    from local_rag_backend.infrastructure.persistence.sql.base import Base

    engine = create_engine(settings.sqlite_url, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    doc_repo = SqlDocumentStorage(session_factory=SessionLocal)
    docs = doc_repo.get_all_documents()

    # Should have multiple chunks due to small chunk size
    assert len(docs) > 1
    from local_rag_backend.core.services.chunking import chunk_chars_v1

    expected = [
        chunk.text
        for chunk in chunk_chars_v1(f"Long Question\n\n{long_content}", max_chars=50, overlap=10)
    ]
    docs_by_index = sorted(docs, key=lambda doc: int((doc.metadata or {})["chunk_index"]))
    assert [doc.content for doc in docs_by_index] == expected


def test_bootstrap_fails_when_configured_csv_is_missing(tmp_path, monkeypatch):
    """Bootstrap must fail fast when FAQ_CSV points to a missing file."""
    monkeypatch.setattr(settings, "faq_csv", "nonexistent.csv", raising=False)
    monkeypatch.setattr(settings, "csv_has_header", True, raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "sqlite_url", f"sqlite:///{tmp_path}/app.db", raising=False)

    with pytest.raises(FileNotFoundError, match="FAQ_CSV must point to an existing CSV file"):
        run_sample_data_ingestion()

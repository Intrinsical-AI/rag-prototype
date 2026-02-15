import csv
import importlib
from unittest.mock import patch

from local_rag_backend.core.domain.entities import LoadedItem
from local_rag_backend.settings import settings


def test_bootstrap_with_ingestion_pipeline_sparse_mode(tmp_path, monkeypatch, capsys):
    """Test bootstrap using ingestion pipeline in sparse mode."""
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

    from local_rag_backend.scripts import bootstrap

    importlib.reload(bootstrap)
    bootstrap.main(settings=settings)

    captured = capsys.readouterr()
    assert "Ingested" in captured.out
    assert "sparse mode" in captured.out

    # Verify documents were stored
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from local_rag_backend.infrastructure.persistence.sqlalchemy.base import Base
    from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

    engine = create_engine(settings.sqlite_url, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    doc_repo = SqlDocumentStorage(session_factory=SessionLocal)
    docs = doc_repo.get_all_documents()

    assert len(docs) == 2
    assert any("RAG" in doc.content for doc in docs)
    assert any("funciona" in doc.content for doc in docs)


def test_bootstrap_with_ingestion_pipeline_dense_mode(tmp_path, monkeypatch, capsys):
    """Test bootstrap using ingestion pipeline in dense mode."""

    class DummyEmbedder:
        dim = 4

        def embed(self, texts):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    class DummyFaissIndex:
        def __init__(self, index_path, id_map_path, dim=None):
            self.index_path = index_path
            self.id_map_path = id_map_path
            self.dim = 4
            self.id_map = []

        def add_to_index(self, ids, vecs):
            self.id_map.extend(ids)

        def search(self, q, k):
            return ([0], [0.9])

        def save(self):
            pass

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
        "local_rag_backend.infrastructure.embeddings.sentence_transformers.SentenceTransformerEmbedder",
        lambda model_name=None: DummyEmbedder(),
        raising=True,
    )
    monkeypatch.setattr(
        "local_rag_backend.infrastructure.persistence.faiss.faiss_.FaissIndex",
        DummyFaissIndex,
    )

    from local_rag_backend.scripts import bootstrap

    importlib.reload(bootstrap)
    bootstrap.main(settings=settings)

    captured = capsys.readouterr()
    assert "Ingested" in captured.out
    assert "SQL and FAISS" in captured.out


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

    from local_rag_backend.scripts import bootstrap

    importlib.reload(bootstrap)
    bootstrap.main(settings=settings)

    # Verify multiple chunks were created
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from local_rag_backend.infrastructure.persistence.sqlalchemy.base import Base
    from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

    engine = create_engine(settings.sqlite_url, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    doc_repo = SqlDocumentStorage(session_factory=SessionLocal)
    docs = doc_repo.get_all_documents()

    # Should have multiple chunks due to small chunk size
    assert len(docs) > 1
    # Each chunk should contain the title metadata
    assert all("Long Question" in doc.content for doc in docs)


def test_bootstrap_with_repo_csv_fallback(tmp_path, monkeypatch, capsys):
    """Test bootstrap falls back to the repo CSV when the configured file doesn't exist."""

    # Mock CSV loader to return test data (regardless of which fallback path is used)
    def mock_csv_loader_load(self):
        return iter(
            [
                LoadedItem(
                    text="Packaged Question\n\nPackaged Answer",
                    metadata={"title": "Packaged Question"},
                )
            ]
        )

    # Configure non-existent local CSV
    monkeypatch.setattr(settings, "faq_csv", "nonexistent.csv", raising=False)
    monkeypatch.setattr(settings, "csv_has_header", True, raising=False)
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "sqlite_url", f"sqlite:///{tmp_path}/app.db", raising=False)

    # Mock CSVLoader to simulate fallback data
    with patch(
        "local_rag_backend.infrastructure.ingestion.loaders.csv_loader.CSVLoader.load",
        mock_csv_loader_load,
    ):
        from local_rag_backend.scripts import bootstrap

        importlib.reload(bootstrap)
        bootstrap.main()

        captured = capsys.readouterr()
        assert "Ingested" in captured.out

        # Verify fallback data was loaded
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from local_rag_backend.infrastructure.persistence.sqlalchemy.base import Base
        from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

        engine = create_engine(settings.sqlite_url, connect_args={"check_same_thread": False})
        SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        Base.metadata.create_all(bind=engine)

        doc_repo = SqlDocumentStorage(session_factory=SessionLocal)
        docs = doc_repo.get_all_documents()

        assert len(docs) == 1
        assert "Packaged Question" in docs[0].content

import csv

from local_rag_backend.scripts.sample_data_ingestion import run_sample_data_ingestion
from local_rag_backend.settings import settings


def test_bootstrap_ingests_data(tmp_path, monkeypatch, caplog):
    class DummyEmbedder:
        dim = 4

        def embed(self, texts):
            return [[0.0] * self.dim for _ in texts]

    # Mock FAISS Index to ensure consistent dim
    class DummyVectorIndex:
        def __init__(self, index_path, id_map_path, dim=None):  # <-- Aquí el cambio
            self.index_path = index_path
            self.id_map_path = id_map_path
            self.dim = 4  # same as DummyEmbedder
            self.id_map = []

        def add_to_index(self, ids, vecs):
            self.id_map.extend(ids)

        def search(self, q, k):
            return ([0], [0.0])

        def save(self):
            pass

    monkeypatch.setattr(settings, "st_embedding_model", "dummy-4", raising=False)
    monkeypatch.setattr(
        "local_rag_backend.infrastructure.embeddings.sentence_transformers.SentenceTransformerEmbedder",
        lambda model_name=None: DummyEmbedder(),
        raising=True,
    )
    # Patch the class in the module where VectorStorage uses it
    monkeypatch.setattr(
        "local_rag_backend.infrastructure.persistence.vector.storage.VectorIndex",
        DummyVectorIndex,
    )

    # Temporary CSV and settings
    csv_file = tmp_path / "faq.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["Q", "A"])
        writer.writerow(["¿Qué es RAG?", "Es Retrieval-Augmented Generation."])

    monkeypatch.setattr(settings, "faq_csv", str(csv_file), raising=False)
    monkeypatch.setattr(settings, "csv_has_header", True, raising=False)
    monkeypatch.setattr(settings, "index_path", str(tmp_path / "idx.faiss"), raising=False)
    monkeypatch.setattr(settings, "id_map_path", str(tmp_path / "id.json"), raising=False)
    monkeypatch.setattr(settings, "sqlite_url", f"sqlite:///{tmp_path}/app.db", raising=False)

    # Reset singleton if needed
    try:
        from local_rag_backend.composition import factory

        if hasattr(factory, "reset_rag_service"):
            factory.reset_rag_service()
    except ImportError:
        pass

    import logging

    with caplog.at_level(logging.INFO):
        run_sample_data_ingestion()

    assert any("Ingested" in m or "Ingerido" in m for m in caplog.messages)

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from local_rag_backend.infrastructure.persistence.sql.alchemy_engine import SqlDocumentStorage

    engine = create_engine(settings.sqlite_url)
    Session = sessionmaker(bind=engine)
    doc_storage = SqlDocumentStorage(session_factory=Session)

    docs = doc_storage.get_all_documents()
    assert len(docs) == 1 and "RAG" in docs[0].content

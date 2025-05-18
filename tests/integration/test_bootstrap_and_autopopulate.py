import csv
import importlib

from src.settings import settings


def test_bootstrap_ingests_data(tmp_path, monkeypatch, capsys):
    class DummyEmbedder:
        dim = 4

        def embed(self, texts):
            return [[0.0] * self.dim for _ in texts]

    # Mock FAISS Index para asegurar dim consistente
    from src.infrastructure.persistence.faiss import index as faiss_index_mod

    class DummyFaissIndex:
        def __init__(self, index_path, id_map_path, dim=None):  # <-- Aquí el cambio
            self.index_path = index_path
            self.id_map_path = id_map_path
            self.dim = 4  # igual a DummyEmbedder
            self.id_map = []

        def add_to_index(self, ids, vecs):
            self.id_map.extend(ids)

        def search(self, q, k):
            return ([0], [0.0])

        def save(self):
            pass

    monkeypatch.setattr(settings, "st_embedding_model", "dummy-4", raising=False)
    monkeypatch.setattr(
        "src.infrastructure.embeddings.sentence_transformers.SentenceTransformerEmbedder",
        lambda model_name=None: DummyEmbedder(),
        raising=True,
    )
    monkeypatch.setattr(faiss_index_mod, "FaissIndex", DummyFaissIndex)

    # CSV temporal y settings
    csv_file = tmp_path / "faq.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["Q", "A"])
        writer.writerow(["¿Qué es RAG?", "Es Retrieval-Augmented Generation."])

    monkeypatch.setattr(settings, "faq_csv", str(csv_file), raising=False)
    monkeypatch.setattr(settings, "csv_has_header", True, raising=False)
    monkeypatch.setattr(
        settings, "index_path", str(tmp_path / "idx.faiss"), raising=False
    )
    monkeypatch.setattr(
        settings, "id_map_path", str(tmp_path / "id.pkl"), raising=False
    )
    monkeypatch.setattr(
        settings, "sqlite_url", f"sqlite:///{tmp_path}/app.db", raising=False
    )

    # Resetear singleton si hace falta
    try:
        from src.app import factory

        if hasattr(factory, "reset_rag_service"):
            factory.reset_rag_service()
    except ImportError:
        pass

    import scripts.bootstrap as bootstrap

    importlib.reload(bootstrap)
    bootstrap.main()

    captured = capsys.readouterr()
    assert "Ingested" in captured.out or "Ingerido" in captured.out

    from src.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage

    docs = SqlDocumentStorage().get_all_documents()
    assert len(docs) == 1 and "RAG" in docs[0].content

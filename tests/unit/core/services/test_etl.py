import pytest

from src.core.services.etl import ETLService


class DummyDocRepo:
    def __init__(self):
        self.saved = []
        self.next_id = 1

    def store_documents(self, texts):
        # Simula devolver IDs únicos por orden de textos
        ids = list(range(self.next_id, self.next_id + len(texts)))
        self.saved.extend(zip(ids, texts))
        self.next_id += len(texts)
        return ids


class DummyEmbedder:
    def __init__(self):
        self.calls = []
        self._counter = 0

    def embed(self, texts):
        self.calls.append(list(texts))
        # Un embedding “dummy” por texto: [n, n, n...]
        return [[i, i + 1, i + 2] for i in range(len(texts))]


class DummyVectorRepo:
    def __init__(self):
        self.upserts = []

    def upsert(self, ids, embeddings):
        # Guarda para comprobación
        self.upserts.append((list(ids), list(embeddings)))


def test_etl_ingest_happy_path():
    doc_repo = DummyDocRepo()
    embedder = DummyEmbedder()
    vector_repo = DummyVectorRepo()

    etl = ETLService(doc_repo, vector_repo, embedder)
    texts = ["Primero", "Segundo"]
    ids = etl.ingest(texts)

    # Se almacenan los textos
    assert [t for (_, t) in doc_repo.saved] == texts
    # Embeddings se generan sobre mismos textos
    assert embedder.calls[0] == texts
    # VectorRepo recibe los mismos ids y embeddings
    assert vector_repo.upserts[0][0] == ids
    # Mismos length
    assert len(ids) == 2
    assert all(isinstance(i, int) for i in ids)


def test_etl_empty_input():
    doc_repo = DummyDocRepo()
    embedder = DummyEmbedder()
    vector_repo = DummyVectorRepo()
    etl = ETLService(doc_repo, vector_repo, embedder)
    ids = etl.ingest([])
    # Nada guardado ni generado
    assert ids == []
    assert doc_repo.saved == []
    assert embedder.calls == [[]] or embedder.calls == []  # Según implementación
    assert vector_repo.upserts == [] or vector_repo.upserts == [([], [])]


def test_etl_error_propagation_on_docrepo():
    class FailingDocRepo(DummyDocRepo):
        def store_documents(self, texts):
            raise RuntimeError("fail-doc")

    doc_repo = FailingDocRepo()
    embedder = DummyEmbedder()
    vector_repo = DummyVectorRepo()
    etl = ETLService(doc_repo, vector_repo, embedder)
    with pytest.raises(RuntimeError, match="fail-doc"):
        etl.ingest(["X"])
    # Nada debería haberse almacenado
    assert embedder.calls == []
    assert vector_repo.upserts == []


def test_etl_error_propagation_on_embedder():
    class FailingEmbedder(DummyEmbedder):
        def embed(self, texts):
            raise RuntimeError("fail-embed")

    doc_repo = DummyDocRepo()
    embedder = FailingEmbedder()
    vector_repo = DummyVectorRepo()
    etl = ETLService(doc_repo, vector_repo, embedder)
    with pytest.raises(RuntimeError, match="fail-embed"):
        etl.ingest(["Y"])
    # Documentos sí guardados, embeddings no, vector store tampoco
    assert [t for (_, t) in doc_repo.saved] == ["Y"]
    assert vector_repo.upserts == []


def test_etl_error_propagation_on_vectorstore():
    class FailingVectorRepo(DummyVectorRepo):
        def upsert(self, ids, vectors):
            raise RuntimeError("fail-vector")

    doc_repo = DummyDocRepo()
    embedder = DummyEmbedder()
    vector_repo = FailingVectorRepo()
    etl = ETLService(doc_repo, vector_repo, embedder)
    with pytest.raises(RuntimeError, match="fail-vector"):
        etl.ingest(["Z"])
    # Doc y embeddings generados
    assert [t for (_, t) in doc_repo.saved] == ["Z"]
    assert embedder.calls and "Z" in embedder.calls[0]


def test_etl_handles_duplicates():
    doc_repo = DummyDocRepo()
    embedder = DummyEmbedder()
    vector_repo = DummyVectorRepo()
    etl = ETLService(doc_repo, vector_repo, embedder)
    texts = ["A", "A", "B"]
    ids = etl.ingest(texts)
    # Debe devolver 3 ids distintos (aunque textos repetidos)
    assert len(set(ids)) == 3
    assert [t for (_, t) in doc_repo.saved] == texts

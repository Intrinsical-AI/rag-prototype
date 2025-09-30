import pytest

from local_rag_backend.core.services.etl import ETLService


class DummyDocRepo:
    def __init__(self):
        self.saved = []
        self.next_id = 1

    def store_documents(self, texts):
        """Simula devolver IDs únicos por orden de textos"""
        ids = list(range(self.next_id, self.next_id + len(texts)))
        self.saved.extend(zip(ids, texts, strict=False))
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


@pytest.mark.parametrize(
    "texts",
    [
        ["A"],
        ["  trim  ", "   \t"],  # whitespace present (ETL stores raw; trimming happens at API)
        ["áéíóú", "漢字"],  # unicode inputs
        ["X" * 10000],  # very long text
        ["dup", "dup", "unique"],
    ],
)
def test_etl_ingest_various_inputs(texts):
    doc_repo = DummyDocRepo()
    embedder = DummyEmbedder()
    vector_repo = DummyVectorRepo()
    etl = ETLService(doc_repo, vector_repo, embedder)

    ids = etl.ingest(texts)

    assert isinstance(ids, list)
    assert len(ids) == len(texts)
    assert all(isinstance(i, int) for i in ids)
    # Order preserved at storage
    assert [t for (_, t) in doc_repo.saved] == texts
    # Embedder called with same texts
    assert embedder.calls and embedder.calls[0] == texts


def test_etl_empty_input():
    doc_repo = DummyDocRepo()
    embedder = DummyEmbedder()
    vector_repo = DummyVectorRepo()
    etl = ETLService(doc_repo, vector_repo, embedder)
    ids = etl.ingest([])
    # Nothing stored or generated
    assert ids == []
    assert doc_repo.saved == []
    assert embedder.calls == [[]] or embedder.calls == []  # According to implementation
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
    # Nothing should have been stored
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
    # Documents stored, embeddings not generated, vector store not updated
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
    # Documents stored, embeddings generated, vector store not updated
    assert [t for (_, t) in doc_repo.saved] == ["Z"]
    assert embedder.calls and "Z" in embedder.calls[0]


def test_etl_handles_duplicates():
    doc_repo = DummyDocRepo()
    embedder = DummyEmbedder()
    vector_repo = DummyVectorRepo()
    etl = ETLService(doc_repo, vector_repo, embedder)
    texts = ["A", "A", "B"]
    ids = etl.ingest(texts)
    # Should return 3 distinct ids (even with repeated texts)
    assert len(set(ids)) == 3
    assert [t for (_, t) in doc_repo.saved] == texts

# tests/unit/app/test_dependencies.py
from types import SimpleNamespace

from local_rag_backend.app import dependencies as deps, factory
from local_rag_backend.core.domain.entities import Document
from local_rag_backend.settings import settings


def _reset_cache() -> None:
    deps.reset_rag_service()


async def test_get_rag_service_sparse_openai(monkeypatch):
    _reset_cache()
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", False, raising=False)

    # Minimal dummies
    class DummyDocRepo:
        def get_all_documents(self):
            return [Document(id=1, content="doc1"), Document(id=2, content="doc2")]

    monkeypatch.setattr(factory, "SqlDocumentStorage", lambda *a, **k: DummyDocRepo())
    monkeypatch.setattr(factory, "SparseBM25Retriever", lambda **k: SimpleNamespace())

    class DummyRS:
        def __init__(self, retriever, generator, history_storage):
            self.retriever = retriever
            self.generator = generator
            self.history_storage = history_storage

    monkeypatch.setattr(factory, "RagService", DummyRS)
    monkeypatch.setattr(factory, "OpenAIGenerator", lambda *a, **k: SimpleNamespace())

    svc1 = await deps.get_rag_service()
    svc2 = await deps.get_rag_service()
    assert svc1 is svc2  # cached
    assert hasattr(svc1, "retriever")


async def test_get_rag_service_dense_ollama(monkeypatch):
    _reset_cache()
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", True, raising=False)

    class DummyEmbedder:
        dim = 4

    monkeypatch.setattr(factory, "SentenceTransformerEmbedder", lambda **k: DummyEmbedder())
    monkeypatch.setattr(factory, "FaissVectorStorage", lambda **k: SimpleNamespace())
    monkeypatch.setattr(factory, "DenseFaissRetriever", lambda **k: SimpleNamespace())
    monkeypatch.setattr(factory, "OllamaGenerator", lambda *a, **k: SimpleNamespace())

    class DummyRS:
        def __init__(self, retriever, generator, history_storage):
            self.retriever = retriever
            self.generator = generator
            self.history_storage = history_storage

    monkeypatch.setattr(factory, "RagService", DummyRS)

    svc = await deps.get_rag_service()
    assert hasattr(svc, "retriever")


async def test_get_rag_service_hybrid_openai(monkeypatch):
    _reset_cache()
    monkeypatch.setattr(settings, "retrieval_mode", "hybrid", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", False, raising=False)

    class DummyDocRepo:
        def get_all_documents(self):
            return [Document(id=1, content="doc1"), Document(id=2, content="doc2")]

    monkeypatch.setattr(factory, "SqlDocumentStorage", lambda *a, **k: DummyDocRepo())

    class DummyEmbedder:
        dim = 4

    monkeypatch.setattr(factory, "SentenceTransformerEmbedder", lambda **k: DummyEmbedder())
    monkeypatch.setattr(factory, "FaissVectorStorage", lambda **k: SimpleNamespace())
    monkeypatch.setattr(factory, "DenseFaissRetriever", lambda **k: SimpleNamespace())
    monkeypatch.setattr(factory, "HybridRetriever", lambda **k: SimpleNamespace())
    monkeypatch.setattr(factory, "OpenAIGenerator", lambda *a, **k: SimpleNamespace())

    class DummyRS:
        def __init__(self, retriever, generator, history_storage):
            self.retriever = retriever
            self.generator = generator
            self.history_storage = history_storage

    monkeypatch.setattr(factory, "RagService", DummyRS)

    svc = await deps.get_rag_service()
    assert hasattr(svc, "retriever")


async def test_ingest_docs_resets_cached_rag_service(asgi_client, in_memory_sqlite, monkeypatch):
    _reset_cache()
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", False, raising=False)

    calls: list[object] = []

    def _build():
        obj = object()
        calls.append(obj)
        return obj

    monkeypatch.setattr(factory, "build_rag_service", _build, raising=True)

    svc1 = await deps.get_rag_service()
    assert svc1 is calls[0]

    r = await asgi_client.post("/api/docs", json={"texts": ["hello"]})
    assert r.status_code == 200
    assert r.json()["count"] == 1

    svc2 = await deps.get_rag_service()
    assert svc2 is calls[1]
    assert svc2 is not svc1


def test_cached_rag_service_cache_does_not_accumulate_on_token_changes(monkeypatch):
    # Simulate cross-process invalidation: token changes without calling reset_rag_service()
    factory._get_cached_rag_service.cache_clear()

    monkeypatch.setattr(factory, "build_rag_service", lambda: object(), raising=True)
    _ = factory._get_cached_rag_service("t1")
    _ = factory._get_cached_rag_service("t2")
    _ = factory._get_cached_rag_service("t3")

    info = factory._get_cached_rag_service.cache_info()
    assert info.maxsize == 1
    assert info.currsize == 1

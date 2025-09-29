# tests/unit/app/test_dependencies.py
from types import SimpleNamespace

from local_rag_backend.app import dependencies as deps
from local_rag_backend.settings import settings


def _reset_cache():
    try:
        deps.get_rag_service.cache_clear()
    except Exception:
        pass


def test_get_rag_service_sparse_openai(monkeypatch):
    _reset_cache()
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", False, raising=False)

    # Minimal dummies
    monkeypatch.setattr(deps, "get_corpus_and_ids", lambda *a, **k: (["doc1", "doc2"], [1, 2]))
    monkeypatch.setattr(deps, "SparseBM25Retriever", lambda **k: SimpleNamespace())

    class DummyRS:
        def __init__(self, retriever, generator, history_storage):
            self.retriever = retriever
            self.generator = generator
            self.history_storage = history_storage

    monkeypatch.setattr(deps, "RagService", DummyRS)
    monkeypatch.setattr(deps, "OpenAIGenerator", lambda *a, **k: SimpleNamespace())

    svc1 = deps.get_rag_service()
    svc2 = deps.get_rag_service()
    assert svc1 is svc2  # cached
    assert hasattr(svc1, "retriever")


def test_get_rag_service_dense_ollama(monkeypatch):
    _reset_cache()
    monkeypatch.setattr(settings, "retrieval_mode", "dense", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", True, raising=False)

    class DummyEmbedder:
        dim = 4

    monkeypatch.setattr(deps, "SentenceTransformerEmbedder", lambda **k: DummyEmbedder())
    monkeypatch.setattr(deps, "FaissVectorStorage", lambda **k: SimpleNamespace())
    monkeypatch.setattr(deps, "DenseFaissRetriever", lambda **k: SimpleNamespace())
    monkeypatch.setattr(deps, "OllamaGenerator", lambda *a, **k: SimpleNamespace())

    class DummyRS:
        def __init__(self, retriever, generator, history_storage):
            self.retriever = retriever
            self.generator = generator
            self.history_storage = history_storage

    monkeypatch.setattr(deps, "RagService", DummyRS)

    svc = deps.get_rag_service()
    assert hasattr(svc, "retriever")


def test_get_rag_service_hybrid_openai(monkeypatch):
    _reset_cache()
    monkeypatch.setattr(settings, "retrieval_mode", "hybrid", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(settings, "ollama_enabled", False, raising=False)

    class DummyEmbedder:
        dim = 4

    monkeypatch.setattr(deps, "SentenceTransformerEmbedder", lambda **k: DummyEmbedder())
    monkeypatch.setattr(deps, "FaissVectorStorage", lambda **k: SimpleNamespace())
    monkeypatch.setattr(deps, "DenseFaissRetriever", lambda **k: SimpleNamespace())
    monkeypatch.setattr(deps, "HybridRetriever", lambda **k: SimpleNamespace())
    monkeypatch.setattr(deps, "OpenAIGenerator", lambda *a, **k: SimpleNamespace())

    class DummyRS:
        def __init__(self, retriever, generator, history_storage):
            self.retriever = retriever
            self.generator = generator
            self.history_storage = history_storage

    monkeypatch.setattr(deps, "RagService", DummyRS)

    svc = deps.get_rag_service()
    assert hasattr(svc, "retriever")

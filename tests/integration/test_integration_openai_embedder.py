import pytest

from local_rag_backend.infrastructure.embeddings import openai as openai_embedder_mod
from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder
from local_rag_backend.settings import settings


class _DummyEmbeddingItem:
    def __init__(self, embedding):
        self.embedding = embedding


class _DummyEmbeddingsResp:
    def __init__(self, vectors):
        self.data = [_DummyEmbeddingItem(v) for v in vectors]


class _DummyEmbeddingsAPI:
    def __init__(self, vectors_by_text):
        self._vectors_by_text = vectors_by_text

    def create(self, model, input):
        vectors = [self._vectors_by_text[t] for t in input]
        return _DummyEmbeddingsResp(vectors)


class _DummyOpenAI:
    def __init__(self, api_key):
        self._api_key = api_key
        self.embeddings = _DummyEmbeddingsAPI(
            {
                "a": [0.0, 0.0, 0.0, 0.0],
                "b": [1.0, 0.0, 0.0, 0.0],
            }
        )


@pytest.mark.integration
def test_openai_embedder_returns_vectors(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(settings, "openai_embedding_model", "dummy-4", raising=False)
    monkeypatch.setattr(openai_embedder_mod, "_MODEL_DIM", {"dummy-4": 4}, raising=False)
    monkeypatch.setattr(openai_embedder_mod, "OpenAI", _DummyOpenAI, raising=True)

    emb = OpenAIEmbedder(model="dummy-4")
    out = emb.embed(["a", "b"])
    assert isinstance(out, list)
    assert len(out) == 2
    assert emb.dim == 4
    assert out[0] == [0.0, 0.0, 0.0, 0.0]
    assert out[1] == [1.0, 0.0, 0.0, 0.0]


@pytest.mark.integration
def test_openai_embedder_empty_input(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(settings, "openai_embedding_model", "dummy-4", raising=False)
    monkeypatch.setattr(openai_embedder_mod, "_MODEL_DIM", {"dummy-4": 4}, raising=False)

    class _NoCallEmbeddingsAPI:
        def create(self, **_):  # pragma: no cover
            raise AssertionError("embeddings.create() should not be called for empty input")

    class _NoCallOpenAI:
        def __init__(self, api_key):
            self.embeddings = _NoCallEmbeddingsAPI()

    monkeypatch.setattr(openai_embedder_mod, "OpenAI", _NoCallOpenAI, raising=True)

    emb = OpenAIEmbedder(model="dummy-4")
    assert emb.embed([]) == []

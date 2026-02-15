import pytest

from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder
from local_rag_backend.settings import settings


def test_openai_embedder_requires_api_key(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIEmbedder(model="text-embedding-3-small")


import pytest

from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder
from local_rag_backend.settings import Settings


def test_openai_embedder_requires_api_key(monkeypatch):
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIEmbedder(model="text-embedding-3-small", settings_obj=Settings(openai_api_key=None))


def test_openai_embedder_passes_configured_timeout(monkeypatch):
    captured: dict[str, object] = {}

    class DummyClient:
        class embeddings:
            @staticmethod
            def create(**kwargs):
                return type("Resp", (), {"data": []})()

    def _dummy_openai(**kwargs):
        captured.update(kwargs)
        return DummyClient()

    monkeypatch.setattr("local_rag_backend.infrastructure.embeddings.openai.OpenAI", _dummy_openai)

    OpenAIEmbedder(
        model="text-embedding-3-small",
        settings_obj=Settings(openai_api_key="k", openai_request_timeout=17),
    )
    assert captured.get("timeout") == 17

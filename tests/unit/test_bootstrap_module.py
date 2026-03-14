import pytest


@pytest.mark.unit
async def test_bootstrap_rag_service_builds(monkeypatch, in_memory_sqlite, reset_app_context):
    from local_rag_backend import bootstrap
    from local_rag_backend.composition import factory
    from local_rag_backend.settings import settings

    # Ensure generator selection is deterministic and doesn't hit network.
    class DummyGen:
        def generate(self, question, contexts):
            return "ok"

    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(factory, "OpenAIGenerator", lambda **_k: DummyGen())

    _ = reset_app_context
    svc = bootstrap.bootstrap_rag_service()
    assert hasattr(svc, "ask")
    resp = svc.ask("hello", top_k=1)
    assert "answer" in resp

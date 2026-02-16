# tests/unit/infrastructure/llms/test_openai_generator.py

import pytest
from fastapi import HTTPException

from local_rag_backend.infrastructure.llms.openai_chat import OpenAIGenerator
from local_rag_backend.settings import settings


# --------------------------------------------------------------------------- #
def make_dummy_openai(should_raise=False):
    class DummyComp:
        def create(self, **_):
            if should_raise:
                # Raise a generic exception; generator should map it to HTTP 502
                raise Exception("boom")

            class DummyResp:
                choices = [type("Msg", (), {"message": type("Cont", (), {"content": "OK"})()})]

            return DummyResp()

    class DummyChat:
        completions = DummyComp()

    class DummyClient:
        chat = DummyChat()

    return lambda *a, **k: DummyClient()


# --------------------------------------------------------------------------- #


def test_generate_success(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "DUMMY", raising=False)

    monkeypatch.setattr(
        "local_rag_backend.infrastructure.llms.openai_chat.OpenAI", make_dummy_openai()
    )
    gen = OpenAIGenerator()
    out = gen.generate("hola", ["ctx"])
    assert out == "OK"


def test_generate_api_error(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "DUMMY", raising=False)

    monkeypatch.setattr(
        "local_rag_backend.infrastructure.llms.openai_chat.OpenAI",
        make_dummy_openai(should_raise=True),
    )
    gen = OpenAIGenerator()
    with pytest.raises(HTTPException) as exc:
        gen.generate("fallará", ["ctx"])
    assert exc.value.status_code == 502


def test_generator_requires_api_key(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIGenerator()


def test_generator_passes_configured_timeout(monkeypatch):
    captured: dict[str, object] = {}

    class DummyClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return type(
                        "Resp",
                        (),
                        {
                            "choices": [
                                type(
                                    "Choice",
                                    (),
                                    {"message": type("Msg", (), {"content": "ok"})()},
                                )()
                            ]
                        },
                    )()

    def _dummy_openai(**kwargs):
        captured.update(kwargs)
        return DummyClient()

    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(settings, "openai_request_timeout", 23, raising=False)
    monkeypatch.setattr("local_rag_backend.infrastructure.llms.openai_chat.OpenAI", _dummy_openai)

    gen = OpenAIGenerator()
    out = gen.generate("q", ["ctx"])
    assert out == "ok"
    assert captured.get("timeout") == 23

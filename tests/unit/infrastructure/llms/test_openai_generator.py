# tests/unit/infrastructure/llms/test_openai_generator.py

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError

from local_rag_backend.core.errors import (
    LLMConfigurationError,
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
)
from local_rag_backend.infrastructure.llms.openai_chat import OpenAIGenerator
from local_rag_backend.settings import get_settings

settings = get_settings()


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
    gen = OpenAIGenerator(settings_obj=settings)
    out = gen.generate("hola", ["ctx"])
    assert out == "OK"


def test_generate_api_error(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "DUMMY", raising=False)

    monkeypatch.setattr(
        "local_rag_backend.infrastructure.llms.openai_chat.OpenAI",
        make_dummy_openai(should_raise=True),
    )
    gen = OpenAIGenerator(settings_obj=settings)
    with pytest.raises(LLMResponseError) as exc:
        gen.generate("fallará", ["ctx"])
    assert "OpenAI API error" in str(exc.value)


def test_generator_requires_api_key(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)
    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY"):
        OpenAIGenerator(settings_obj=settings)


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

    gen = OpenAIGenerator(settings_obj=settings)
    out = gen.generate("q", ["ctx"])
    assert out == "ok"
    assert captured.get("timeout") == 23


@pytest.mark.parametrize(
    ("provider_error", "expected_error"),
    [
        (APITimeoutError(request=httpx.Request("POST", "https://example.test")), LLMTimeoutError),
        (
            APIConnectionError(request=httpx.Request("POST", "https://example.test")),
            LLMConnectionError,
        ),
    ],
)
def test_generator_preserves_timeout_and_connection_error_types(
    monkeypatch, provider_error, expected_error
):
    def fail_create(**_kwargs):
        raise provider_error

    class DummyClient:
        class chat:
            class completions:
                create = staticmethod(fail_create)

    monkeypatch.setattr(settings, "openai_api_key", "DUMMY", raising=False)
    monkeypatch.setattr(
        "local_rag_backend.infrastructure.llms.openai_chat.OpenAI", lambda **_: DummyClient()
    )
    generator = OpenAIGenerator(settings_obj=settings)

    with pytest.raises(expected_error):
        generator.generate("question", ["context"])

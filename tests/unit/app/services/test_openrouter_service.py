from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_rag_backend.app.services import openrouter as service
from local_rag_backend.core.errors import LLMResponseError


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        openrouter_site_url="https://example.test",
        openrouter_app_title="rag-test",
        openrouter_api_key="k",
        openrouter_base_url="https://openrouter.test/api/v1",
        openai_request_timeout=17,
        openrouter_model="openai/gpt-4o-mini",
    )


def _payload() -> service.OpenRouterGenerateInput:
    return service.OpenRouterGenerateInput(
        model=None,
        system_instruction="sys",
        user_content="hello",
        temperature=0.2,
        max_tokens=100,
        top_p=0.9,
    )


def test_generate_openrouter_sync_builds_headers_and_defaults_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class DummyUsage:
        prompt_tokens = 3
        completion_tokens = 4
        total_tokens = 7

    class DummyMsg:
        content = "ok"

    class DummyChoice:
        message = DummyMsg()

    class DummyResp:
        choices = [DummyChoice()]
        usage = DummyUsage()

    class DummyClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs: object) -> DummyResp:
                    captured["request"] = kwargs
                    return DummyResp()

    def _fake_create_client(**kwargs: object) -> DummyClient:
        captured["client"] = kwargs
        return DummyClient()

    monkeypatch.setattr(service, "create_openai_client", _fake_create_client, raising=True)

    out = service.generate_openrouter_sync(payload=_payload(), settings_obj=_settings())

    assert out.text == "ok"
    assert out.usage is not None
    assert out.usage.total_tokens == 7

    client_args = captured["client"]
    assert isinstance(client_args, dict)
    assert client_args["default_headers"] == {
        "HTTP-Referer": "https://example.test",
        "X-Title": "rag-test",
    }
    req = captured["request"]
    assert isinstance(req, dict)
    assert req["model"] == "openai/gpt-4o-mini"


def test_generate_openrouter_sync_none_content_becomes_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyMsg:
        content = None

    class DummyChoice:
        message = DummyMsg()

    class DummyResp:
        choices = [DummyChoice()]
        usage = None

    class DummyClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs: object) -> DummyResp:
                    return DummyResp()

    monkeypatch.setattr(
        service,
        "create_openai_client",
        lambda **_kwargs: DummyClient(),
        raising=True,
    )

    out = service.generate_openrouter_sync(payload=_payload(), settings_obj=_settings())
    assert out.text == ""
    assert out.usage is None


def test_generate_openrouter_sync_malformed_content_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyMsg:
        content = ["not-string"]

    class DummyChoice:
        message = DummyMsg()

    class DummyResp:
        choices = [DummyChoice()]
        usage = None

    class DummyClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs: object) -> DummyResp:
                    return DummyResp()

    monkeypatch.setattr(
        service,
        "create_openai_client",
        lambda **_kwargs: DummyClient(),
        raising=True,
    )

    with pytest.raises(LLMResponseError, match="malformed response content"):
        service.generate_openrouter_sync(payload=_payload(), settings_obj=_settings())


def test_generate_openrouter_sync_wraps_client_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(**kwargs: object) -> object:
        raise RuntimeError("sdk failure")

    monkeypatch.setattr(service, "create_openai_client", _boom, raising=True)

    with pytest.raises(LLMResponseError, match="sdk failure"):
        service.generate_openrouter_sync(payload=_payload(), settings_obj=_settings())

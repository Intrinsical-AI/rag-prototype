from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError

from local_rag_backend.composition.adapters import (
    _OpenAICompatibleOpenRouterClient,
    build_generator_from_settings,
)
from local_rag_backend.core.errors import LLMConnectionError, LLMTimeoutError
from local_rag_backend.core.ports import OpenRouterGenerateRequest
from local_rag_backend.settings import Settings


@pytest.mark.parametrize(
    ("sdk_error", "expected_error"),
    [
        (APITimeoutError(request=httpx.Request("POST", "https://example.test")), LLMTimeoutError),
        (
            APIConnectionError(request=httpx.Request("POST", "https://example.test")),
            LLMConnectionError,
        ),
    ],
)
def test_openrouter_adapter_preserves_sdk_transport_failure(sdk_error, expected_error) -> None:
    def fail_create(**_kwargs):
        raise sdk_error

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail_create)))
    adapter = _OpenAICompatibleOpenRouterClient(client=client, default_model="test")
    request = OpenRouterGenerateRequest(
        model=None,
        system_instruction="system",
        user_content="question",
        temperature=0.2,
        max_tokens=32,
        top_p=0.9,
    )

    with pytest.raises(expected_error):
        adapter.generate(request=request)


def test_eval_generator_default_prefers_ollama_even_if_openai_listed_first() -> None:
    ollama = object()
    selected = build_generator_from_settings(
        settings_obj=Settings(openai_api_key="test", ollama_enabled=True),
        available_providers={"openai": "configured", "ollama": "enabled"},
        openai_generator_factory=lambda **_kwargs: object(),
        ollama_generator_factory=lambda **_kwargs: ollama,
    )

    assert selected is ollama

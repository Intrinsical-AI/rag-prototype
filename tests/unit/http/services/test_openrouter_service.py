from __future__ import annotations

import pytest

from local_rag_backend.core.errors import LLMResponseError
from local_rag_backend.core.ports import OpenRouterGenerateRequest, OpenRouterGenerateResult
from local_rag_backend.core.use_cases import openrouter as service


def _payload() -> OpenRouterGenerateRequest:
    return OpenRouterGenerateRequest(
        model=None,
        system_instruction="sys",
        user_content="hello",
        temperature=0.2,
        max_tokens=100,
        top_p=0.9,
    )


def test_generate_openrouter_sync_delegates_to_port() -> None:
    class DummyClient:
        def generate(
            self,
            *,
            request: OpenRouterGenerateRequest,
        ) -> OpenRouterGenerateResult:
            assert request.user_content == "hello"
            return OpenRouterGenerateResult(text="ok")

    out = service.generate_openrouter_sync(payload=_payload(), openrouter_client=DummyClient())
    assert out.text == "ok"


def test_generate_openrouter_sync_wraps_client_errors() -> None:
    class FailingClient:
        def generate(
            self,
            *,
            request: OpenRouterGenerateRequest,
        ) -> OpenRouterGenerateResult:
            raise RuntimeError("sdk failure")

    with pytest.raises(LLMResponseError, match="sdk failure"):
        service.generate_openrouter_sync(payload=_payload(), openrouter_client=FailingClient())

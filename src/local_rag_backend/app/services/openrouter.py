"""Compatibility shim for OpenRouter use-cases (moved to app.application)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_backend.app.application import openrouter as _impl

if TYPE_CHECKING:
    from local_rag_backend.settings import Settings

OpenAI = _impl.OpenAI
create_openai_client = _impl.create_openai_client

OpenRouterGenerateInput = _impl.OpenRouterGenerateInput
OpenRouterGenerateOutput = _impl.OpenRouterGenerateOutput
OpenRouterUsageOut = _impl.OpenRouterUsageOut


def generate_openrouter_sync(
    *,
    payload: OpenRouterGenerateInput,
    settings_obj: Settings,
) -> OpenRouterGenerateOutput:
    return _impl.generate_openrouter_sync(
        payload=payload,
        settings_obj=settings_obj,
        create_openai_client_fn=create_openai_client,
        openai_client_factory=OpenAI,
    )


__all__ = [
    "OpenAI",
    "OpenRouterGenerateInput",
    "OpenRouterGenerateOutput",
    "OpenRouterUsageOut",
    "create_openai_client",
    "generate_openrouter_sync",
]

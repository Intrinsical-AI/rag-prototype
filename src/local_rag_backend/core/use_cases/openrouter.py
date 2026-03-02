"""Application service for OpenRouter proxy calls."""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_backend.core.errors import LLMResponseError
from local_rag_backend.core.ports import (
    OpenRouterGenerateRequest,
    OpenRouterGenerateResult,
)

if TYPE_CHECKING:
    from local_rag_backend.core.ports import OpenRouterClientPort


def generate_openrouter_sync(
    *,
    payload: OpenRouterGenerateRequest,
    openrouter_client: OpenRouterClientPort,
) -> OpenRouterGenerateResult:
    try:
        return openrouter_client.generate(request=payload)
    except Exception as e:
        raise LLMResponseError(f"OpenRouter error: {e!s}") from e

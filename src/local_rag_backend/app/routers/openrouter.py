"""
Bounded router for OpenRouter proxy operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends

from local_rag_backend.app.application.openrouter import (
    OpenRouterGenerateInput,
    OpenRouterGenerateOutput,
    OpenRouterUsageOut,
    generate_openrouter_sync,
)
from local_rag_backend.app.blocking import run_blocking
from local_rag_backend.app.dependencies import get_settings_dependency
from local_rag_backend.app.errors import BadRequestError
from local_rag_backend.app.schemas.openrouter import (
    OpenRouterGenerateRequest,
    OpenRouterGenerateResponse,
    OpenRouterUsage,
)

if TYPE_CHECKING:
    from local_rag_backend.settings import Settings

router = APIRouter()


@router.post(
    "/openrouter/generate",
    response_model=OpenRouterGenerateResponse,
    tags=["LLM"],
    summary="Proxy completion via OpenRouter (OpenAI-compatible)",
)
async def openrouter_generate(
    payload: OpenRouterGenerateRequest,
    settings_obj: Settings = Depends(get_settings_dependency),
) -> OpenRouterGenerateResponse:
    if not (
        getattr(settings_obj, "openrouter_enabled", False)
        and getattr(settings_obj, "openrouter_api_key", None)
    ):
        raise BadRequestError(
            detail="OpenRouter is not configured (set OPENROUTER_ENABLED and OPENROUTER_API_KEY)",
        )

    service_payload = OpenRouterGenerateInput(
        model=payload.model,
        system_instruction=payload.system_instruction,
        user_content=payload.user_content,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        top_p=payload.top_p,
    )
    out = await run_blocking(
        generate_openrouter_sync,
        payload=service_payload,
        settings_obj=settings_obj,
        task_type="network",
    )

    if not isinstance(out, OpenRouterGenerateOutput):
        raise RuntimeError(f"OpenRouter generation returned unexpected type: {type(out).__name__}")
    usage = out.usage
    usage_obj = (
        OpenRouterUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )
        if isinstance(usage, OpenRouterUsageOut)
        else None
    )
    return OpenRouterGenerateResponse(text=out.text, usage=usage_obj)

"""
Bounded router for OpenRouter proxy operations.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from local_rag_backend.app.blocking import run_blocking
from local_rag_backend.app.error_mapping import raise_http_for_runtime_error
from local_rag_backend.app.services.openrouter import (
    OpenRouterGenerateInput,
    OpenRouterGenerateOutput,
    OpenRouterUsageOut,
    generate_openrouter_sync,
)
from local_rag_backend.settings import settings

router = APIRouter()


class OpenRouterUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenRouterGenerateRequest(BaseModel):
    model: str | None = Field(
        default=None, description="OpenRouter model ID, e.g., 'openai/gpt-4o-mini'"
    )
    system_instruction: str = Field(..., min_length=1, max_length=8000)
    user_content: str = Field(..., min_length=1, max_length=8000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)


class OpenRouterGenerateResponse(BaseModel):
    text: str
    usage: OpenRouterUsage | None = None


@router.post(
    "/openrouter/generate",
    response_model=OpenRouterGenerateResponse,
    tags=["LLM"],
    summary="Proxy completion via OpenRouter (OpenAI-compatible)",
)
async def openrouter_generate(payload: OpenRouterGenerateRequest) -> OpenRouterGenerateResponse:
    if not (
        getattr(settings, "openrouter_enabled", False)
        and getattr(settings, "openrouter_api_key", None)
    ):
        raise HTTPException(
            status_code=400,
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
    try:
        out = await run_blocking(
            generate_openrouter_sync,
            payload=service_payload,
            settings_obj=settings,
            task_type="network",
        )
    except Exception as e:
        raise_http_for_runtime_error(e)
        raise

    assert isinstance(out, OpenRouterGenerateOutput)
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


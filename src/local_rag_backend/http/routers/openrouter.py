"""
Bounded router for OpenRouter proxy operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends

from local_rag_backend.core.ports import (
    OpenRouterGenerateRequest as OpenRouterServiceRequest,
    OpenRouterGenerateResult as OpenRouterServiceResult,
)
from local_rag_backend.core.use_cases.errors import BadRequestError
from local_rag_backend.core.use_cases.openrouter import (
    generate_openrouter_sync,
)
from local_rag_backend.http.dependencies import (
    get_app_container_dependency,
    get_settings_dependency,
)
from local_rag_backend.http.schemas.openrouter import (
    OpenRouterGenerateRequest,
    OpenRouterGenerateResponse,
    OpenRouterUsage,
)
from local_rag_backend.infrastructure.concurrency.blocking import run_blocking

if TYPE_CHECKING:
    from local_rag_backend.composition.container import AppContainer
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
    container: AppContainer = Depends(get_app_container_dependency),
    settings_obj: Settings = Depends(get_settings_dependency),
) -> OpenRouterGenerateResponse:
    if not (
        getattr(settings_obj, "openrouter_enabled", False)
        and getattr(settings_obj, "openrouter_api_key", None)
    ):
        raise BadRequestError(
            detail="OpenRouter is not configured (set openrouter_enabled and openrouter_api_key in config.yaml)",
        )

    service_payload = OpenRouterServiceRequest(
        model=payload.model,
        system_instruction=payload.system_instruction,
        user_content=payload.user_content,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        top_p=payload.top_p,
    )

    def _generate_operation() -> OpenRouterServiceResult:
        return generate_openrouter_sync(
            payload=service_payload,
            openrouter_client=container.build_openrouter_client(),
        )

    out = await run_blocking(_generate_operation, task_type="network")
    usage = out.usage
    usage_obj = (
        OpenRouterUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )
        if usage is not None
        else None
    )
    return OpenRouterGenerateResponse(text=out.text, usage=usage_obj)

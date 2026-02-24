"""Application service for OpenRouter proxy calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from openai import OpenAI

from local_rag_backend.core.errors import LLMResponseError
from local_rag_backend.infrastructure.llms.openai_chat import create_openai_client

if TYPE_CHECKING:
    from collections.abc import Callable

    from openai import OpenAI as OpenAIClient

    from local_rag_backend.settings import Settings


@dataclass(frozen=True)
class OpenRouterGenerateInput:
    model: str | None
    system_instruction: str
    user_content: str
    temperature: float | None
    max_tokens: int | None
    top_p: float | None


@dataclass(frozen=True)
class OpenRouterUsageOut:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class OpenRouterGenerateOutput:
    text: str
    usage: OpenRouterUsageOut | None = None


def generate_openrouter_sync(
    *,
    payload: OpenRouterGenerateInput,
    settings_obj: Settings,
    create_openai_client_fn: Callable[..., object] | None = None,
    openai_client_factory: type[OpenAIClient] | None = None,
) -> OpenRouterGenerateOutput:
    create_client = create_openai_client_fn or create_openai_client
    client_factory = openai_client_factory or OpenAI

    headers: dict[str, str] = {}
    if settings_obj.openrouter_site_url is not None:
        headers["HTTP-Referer"] = settings_obj.openrouter_site_url
    if settings_obj.openrouter_app_title is not None:
        headers["X-Title"] = settings_obj.openrouter_app_title

    try:
        client = create_client(
            api_key=settings_obj.openrouter_api_key,
            base_url=settings_obj.openrouter_base_url,
            default_headers=headers or None,
            timeout=settings_obj.openai_request_timeout,
            client_factory=client_factory,
        )
        resp = client.chat.completions.create(
            model=(payload.model or settings_obj.openrouter_model),
            temperature=payload.temperature,
            top_p=payload.top_p,
            max_tokens=payload.max_tokens,
            messages=[
                {"role": "system", "content": payload.system_instruction},
                {"role": "user", "content": payload.user_content},
            ],
        )
    except Exception as e:
        raise LLMResponseError(f"OpenRouter error: {e!s}") from e

    choices = getattr(resp, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise LLMResponseError("OpenRouter error: malformed response (missing choices).")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    content = getattr(message, "content", None)
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    else:
        raise LLMResponseError("OpenRouter error: malformed response content.")

    usage_raw = getattr(resp, "usage", None)
    usage: OpenRouterUsageOut | None = None
    if usage_raw is not None:
        usage = OpenRouterUsageOut(
            prompt_tokens=int(getattr(usage_raw, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage_raw, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage_raw, "total_tokens", 0) or 0),
        )

    return OpenRouterGenerateOutput(text=text, usage=usage)

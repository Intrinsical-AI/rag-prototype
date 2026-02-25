"""OpenRouter bounded-context transport schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


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

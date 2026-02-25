"""Meta/config bounded-context transport schemas."""

from __future__ import annotations

from pydantic import BaseModel


class TemplateResponse(BaseModel):
    name: str
    template: str
    description: str


class ConfigResponse(BaseModel):
    retrieval_mode: str
    hybrid_alpha: float
    temperature: float
    max_tokens: int
    available_providers: list[str]

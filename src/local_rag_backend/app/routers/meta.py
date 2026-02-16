"""
Bounded router for metadata/config endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from local_rag_backend.app.composition import (
    get_available_llm_providers as get_available_llm_providers_from_settings,
)
from local_rag_backend.settings import settings

router = APIRouter()


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


@router.get(
    "/templates",
    response_model=list[TemplateResponse],
    tags=["RAG"],
    summary="Get available prompt templates",
)
async def get_templates() -> list[TemplateResponse]:
    templates = [
        TemplateResponse(
            name="default",
            template=settings.openai_prompt_template,
            description="Default template for OpenAI/OpenRouter models",
        ),
        TemplateResponse(
            name="ollama",
            template=settings.ollama_prompt_template,
            description="Template optimized for Ollama models",
        ),
        TemplateResponse(
            name="concise",
            template="Based on the context below, provide a concise answer.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:",
            description="Concise template for brief responses",
        ),
        TemplateResponse(
            name="detailed",
            template="You are an expert Q&A system. Your task is to answer the user's question based on the provided sources. Synthesize the information from the sources into a coherent, detailed answer.\n\nSources:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            description="Detailed template for comprehensive responses",
        ),
    ]
    return templates


@router.get(
    "/config",
    response_model=ConfigResponse,
    tags=["RAG"],
    summary="Get backend configuration defaults",
)
async def get_config() -> ConfigResponse:
    providers = get_available_llm_providers_from_settings(settings_obj=settings)
    available_providers = list(providers.keys())
    return ConfigResponse(
        retrieval_mode=settings.retrieval_mode,
        hybrid_alpha=settings.hybrid_retrieval_alpha,
        temperature=settings.openai_temperature,
        max_tokens=settings.openai_max_tokens,
        available_providers=available_providers,
    )


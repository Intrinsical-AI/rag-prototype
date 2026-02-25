"""
Bounded router for metadata/config endpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends

from local_rag_backend.composition.adapters import (
    get_available_llm_providers as get_available_llm_providers_from_settings,
)
from local_rag_backend.http.dependencies import get_settings_dependency
from local_rag_backend.http.schemas.meta import ConfigResponse, TemplateResponse

if TYPE_CHECKING:
    from local_rag_backend.settings import Settings

router = APIRouter()


@router.get(
    "/templates",
    response_model=list[TemplateResponse],
    tags=["RAG"],
    summary="Get available prompt templates",
)
async def get_templates(
    settings_obj: Settings = Depends(get_settings_dependency),
) -> list[TemplateResponse]:
    templates = [
        TemplateResponse(
            name="default",
            template=settings_obj.openai_prompt_template,
            description="Default template for OpenAI/OpenRouter models",
        ),
        TemplateResponse(
            name="ollama",
            template=settings_obj.ollama_prompt_template,
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
async def get_config(settings_obj: Settings = Depends(get_settings_dependency)) -> ConfigResponse:
    providers = get_available_llm_providers_from_settings(settings_obj=settings_obj)
    available_providers = list(providers.keys())
    return ConfigResponse(
        retrieval_mode=settings_obj.retrieval_mode,
        hybrid_alpha=settings_obj.hybrid_retrieval_alpha,
        temperature=settings_obj.openai_temperature,
        max_tokens=settings_obj.openai_max_tokens,
        available_providers=available_providers,
    )

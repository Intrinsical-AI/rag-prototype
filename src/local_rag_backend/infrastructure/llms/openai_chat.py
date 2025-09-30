"""
Intrinsical-AI RAG Prototype
Copyright (c) 2025 Intrinsical-AI

Module: OpenAI LLM Generator
Purpose: Cloud-based text generation using OpenAI's Chat Completions API.
         Provides high-quality language model integration with GPT models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from fastapi import HTTPException
from openai import OpenAI

from local_rag_backend.core.ports import GeneratorPort
from local_rag_backend.settings import settings

__all__ = ["OpenAIGenerator"]


class OpenAIGenerator(GeneratorPort):
    """Cloud-based text generator using OpenAI's Chat Completions API.

    This generator provides high-quality text generation using OpenAI's GPT models.
    Supports advanced parameters like temperature, top_p, and custom prompt templates.

    Features:
    - State-of-the-art language models (GPT-4, GPT-3.5)
    - Configurable generation parameters
    - Custom base URLs for OpenAI-compatible APIs
    - Robust error handling and retry logic
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        prompt_template: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        """Initialize OpenAI generator with configuration.

        Args:
            model: OpenAI model name (e.g., 'gpt-4o-mini')
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            prompt_template: Custom prompt template
            api_key: OpenAI API key (defaults to settings)
            base_url: Custom API base URL (for OpenAI-compatible APIs)
            extra_headers: Additional HTTP headers
        """
        self.model = model or settings.openai_model
        self.temperature = temperature if temperature is not None else settings.openai_temperature
        self.top_p = top_p if top_p is not None else settings.openai_top_p
        self.max_tokens = max_tokens if max_tokens is not None else settings.openai_max_tokens
        self.prompt_template = prompt_template or settings.openai_prompt_template

        self.client = OpenAI(
            api_key=(api_key or settings.openai_api_key),
            base_url=base_url,
            default_headers=extra_headers,
        )

    def _build_prompt(self, question: str, contexts: Sequence[str]) -> str:
        """Build the prompt string for the OpenAI API."""
        context_str = "\n".join(f"- {c}" for c in contexts)
        return self.prompt_template.format(context=context_str, question=question)

    def generate(self, question: str, contexts: Sequence[str]) -> str:
        """Generate a response from the OpenAI API."""
        prompt = self._build_prompt(question, contexts)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content
            return content or ""
        except Exception as e:
            # Broadly catch API errors, connection issues, etc.
            raise HTTPException(status_code=502, detail=f"OpenAI API error: {e!s}") from e

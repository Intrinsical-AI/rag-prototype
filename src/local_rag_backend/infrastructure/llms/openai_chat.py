# src/infrastructure/llms/openai_chat.py
"""
OpenAI Chat completion generator (compatible con API v1)

* Instantiated with `OpenAI(api_key=…)`.
* `generate()` builds prompt exactly as expected by asserts.
* Raises typed provider errors; HTTP mapping is handled in app transport layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from openai import OpenAI

from local_rag_backend.core.errors import LLMConfigurationError, LLMResponseError
from local_rag_backend.core.ports import GeneratorPort
from local_rag_backend.core.services.prompting import render_prompt_template
from local_rag_backend.infrastructure.llms.openai_client import create_openai_client
from local_rag_backend.settings import settings

__all__ = ["OpenAIGenerator"]


class OpenAIGenerator(GeneratorPort):
    """Generator using the OpenAI chat completions API."""

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
        resolved_key = api_key or settings.openai_api_key
        if not resolved_key:
            raise LLMConfigurationError("OPENAI_API_KEY is required to use the OpenAI generator.")

        self.model = model or settings.openai_model
        self.temperature = temperature if temperature is not None else settings.openai_temperature
        self.top_p = top_p if top_p is not None else settings.openai_top_p
        self.max_tokens = max_tokens if max_tokens is not None else settings.openai_max_tokens
        self.prompt_template = prompt_template or settings.openai_prompt_template

        self.client = create_openai_client(
            api_key=resolved_key,
            base_url=base_url,
            default_headers=extra_headers,
            timeout=settings.openai_request_timeout,
            client_factory=OpenAI,
        )

    def _build_prompt(self, question: str, contexts: Sequence[str]) -> str:
        """Build the prompt string for the OpenAI API."""
        context_str = "\n".join(f"- {c}" for c in contexts)
        return render_prompt_template(self.prompt_template, context=context_str, question=question)

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
            # Broadly catch provider/runtime SDK errors and map in app layer.
            raise LLMResponseError(f"OpenAI API error: {e!s}") from e

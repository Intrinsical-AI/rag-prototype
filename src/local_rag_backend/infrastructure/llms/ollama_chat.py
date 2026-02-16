# src/adapters/generation/ollama_chat.py
"""
Ollama generator for local Ollama server.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import requests

from local_rag_backend.core.errors import (
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
)
from local_rag_backend.core.ports import GeneratorPort
from local_rag_backend.core.services.prompting import render_prompt_template
from local_rag_backend.settings import (
    settings,  # settings.ollama_base_url y settings.ollama_request_timeout exists
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class OllamaGenerator(GeneratorPort):
    """Generator using a local Ollama server."""

    def __init__(
        self,
        model: str | None = None,
        prompt_template: str | None = None,
        temperature: float | None = None,
    ):
        self.model = model or settings.ollama_model
        self.prompt_template = prompt_template or settings.ollama_prompt_template
        self.temperature = temperature
        self.api_url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"

    def generate(self, question: str, contexts: Sequence[str]) -> str:
        """Generate a response from the Ollama server."""
        context_str = "\n".join(f"- {c}" for c in contexts)
        prompt = render_prompt_template(
            self.prompt_template, context=context_str, question=question
        )

        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if self.temperature is not None:
            payload["options"] = {"temperature": self.temperature}

        try:
            response = requests.post(
                self.api_url, json=payload, timeout=settings.ollama_request_timeout
            )
            response.raise_for_status()
            response_data = response.json()

            if "response" in response_data and isinstance(response_data["response"], str):
                return response_data["response"].strip()
            raise LLMResponseError("Ollama response malformed")

        except requests.exceptions.Timeout as e:
            raise LLMTimeoutError(f"Ollama request timed out to {self.api_url}") from e
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(f"Could not connect to Ollama at {self.api_url}") from e
        except requests.exceptions.RequestException as e:
            status = e.response.status_code if e.response is not None else 500
            detail = e.response.text if e.response is not None else str(e)
            raise LLMResponseError(f"Ollama API error (status={status}): {detail}") from e
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama: {e}", exc_info=True)
            raise LLMResponseError(f"Unexpected error calling Ollama: {e!s}") from e

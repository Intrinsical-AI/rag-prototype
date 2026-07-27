# src/adapters/generation/ollama_chat.py
"""
Ollama generator for local Ollama server.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

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
            response = httpx.post(
                self.api_url, json=payload, timeout=settings.ollama_request_timeout
            )
            response.raise_for_status()
            response_data = response.json()

            if "response" in response_data and isinstance(response_data["response"], str):
                return response_data["response"].strip()
            raise LLMResponseError("Ollama response malformed")

        except httpx.TimeoutException as exc:
            logger.exception("Ollama request timed out")
            raise LLMTimeoutError("Ollama request timed out") from exc
        except httpx.ConnectError as exc:
            logger.exception("Ollama connection failed")
            raise LLMConnectionError("Could not connect to Ollama") from exc
        except httpx.HTTPStatusError as exc:
            logger.exception("Ollama returned an HTTP error")
            raise LLMResponseError("Ollama returned an error response") from exc
        except httpx.RequestError as exc:
            logger.exception("Ollama request failed")
            raise LLMResponseError("Ollama request failed") from exc
        except LLMResponseError:
            raise
        except Exception as exc:
            logger.exception("Unexpected error calling Ollama")
            raise LLMResponseError("Unexpected error calling Ollama") from exc

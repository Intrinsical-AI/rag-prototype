"""
RAG Prototype - Intrinsical-AI (c) 2025
Author: Pablo Pintor
License: MIT

Module: Ollama LLM Generator
Purpose: Local LLM text generation using Ollama server.
         Provides privacy-focused, self-hosted language model integration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import requests
from fastapi import HTTPException

from local_rag_backend.core.ports import GeneratorPort
from local_rag_backend.settings import (
    settings,  # settings.ollama_base_url y settings.ollama_request_timeout exists
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class OllamaGenerator(GeneratorPort):
    """Local LLM text generator using Ollama server.

    This generator provides privacy-focused text generation by connecting to
    a local Ollama server. Supports various open-source models like Llama,
    Gemma, and others available through Ollama.

    Benefits:
    - Complete data privacy (no external API calls)
    - No usage costs after initial setup
    - Customizable models and parameters
    - Offline operation capability
    """

    def __init__(
        self,
        model: str | None = None,
        prompt_template: str | None = None,
        temperature: float | None = None,
    ):
        """Initialize Ollama generator with configuration.

        Args:
            model: Ollama model name (defaults to settings)
            prompt_template: Custom prompt template (defaults to settings)
            temperature: Generation temperature (optional override)
        """
        self.model = model or settings.ollama_model
        self.prompt_template = prompt_template or settings.ollama_prompt_template
        self.temperature = temperature
        self.api_url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"

    def generate(self, question: str, contexts: Sequence[str]) -> str:
        """Generate answer using local Ollama server.

        Args:
            question: User question to answer
            contexts: Retrieved document contexts for grounding

        Returns:
            Generated answer text

        Raises:
            HTTPException: For various Ollama server errors (timeout, connection, etc.)
        """
        context_str = "\n".join(f"- {c}" for c in contexts)
        prompt = self.prompt_template.format(context=context_str, question=question)

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
            raise HTTPException(500, "Ollama response malformed")

        except requests.exceptions.Timeout as e:
            raise HTTPException(504, f"Ollama request timed out to {self.api_url}") from e
        except requests.exceptions.ConnectionError as e:
            raise HTTPException(503, f"Could not connect to Ollama at {self.api_url}") from e
        except requests.exceptions.RequestException as e:
            status = e.response.status_code if e.response is not None else 500
            detail = e.response.text if e.response is not None else str(e)
            raise HTTPException(status, f"Ollama API error: {detail}") from e
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama: {e}", exc_info=True)
            raise HTTPException(500, f"Unexpected error calling Ollama: {e!s}") from e

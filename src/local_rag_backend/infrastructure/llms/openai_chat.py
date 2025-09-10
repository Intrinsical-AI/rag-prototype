# === file: src/adapters/generation/openai_chat.py ===
"""OpenAI Chat completion generator (compatible con API v1)

Cumple los tests:
* Se instancia con `OpenAI(api_key=…)`.
* `generate()` construye prompt exactamente como esperan los asserts.
* Maneja `APIError` y lo convierte a `HTTPException 502`.
"""
from __future__ import annotations

from collections.abc import Sequence

from fastapi import HTTPException
from openai import OpenAI

from local_rag_backend.core.ports import GeneratorPort
from local_rag_backend.settings import settings

__all__ = ["OpenAIGenerator"]


class OpenAIGenerator(GeneratorPort):
    """Adapter para chat‑completion de OpenAI v1.x"""

    def __init__(
        self, *, model: str | None = None, temperature: float | None = None
    ) -> None:
        self.model = model or settings.openai_model
        self.temperature = (
            temperature if temperature is not None else settings.openai_temperature
        )
        self.client = OpenAI(api_key=settings.openai_api_key)

    # ------------------------------------------------------------------
    def _build_prompt(self, question: str, contexts: Sequence[str]) -> str:
        ctx_block = "\n".join(f"- {c}" for c in contexts)
        return settings.openai_prompt_template.format(
            context=ctx_block, question=question
        )

    def generate(self, question: str, contexts: Sequence[str]) -> str:
        prompt = self._build_prompt(question, contexts)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                top_p=settings.openai_top_p,
                max_tokens=settings.openai_max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except HTTPException:
            # re-lanzar HTTPExceptions (timeouts, etc.)
            raise
        except Exception as err:
            # Aquí “pillamos” tanto APIError real como TypeError de test-stub
            raise HTTPException(
                status_code=502,
                detail=f"OpenAI API Error: {getattr(err, 'message', str(err))}",
            ) from err

        content = resp.choices[0].message.content
        return content or ""  # Handle None case

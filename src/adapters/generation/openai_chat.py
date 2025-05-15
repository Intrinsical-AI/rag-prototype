# === file: src/adapters/generation/openai_chat.py ===
"""OpenAI Chat completion generator (compatible con API v1)

Cumple los tests:
* Se instancia con `OpenAI(api_key=…)`.
* `generate()` construye prompt exactamente como esperan los asserts.
* Maneja `APIError` y lo convierte a `HTTPException 502`.
"""
from __future__ import annotations

from typing import List

from fastapi import HTTPException
from openai import OpenAI, APIError  # type: ignore

from src.core.ports import GeneratorPort
from src.settings import settings

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
    def _build_prompt(self, question: str, contexts: List[str]) -> str:
        ctx_block = "\n".join(f"- {c}" for c in contexts)
        return (
            "Answer using ONLY the context provided.\n\n"
            f"CONTEXT:\n{ctx_block}\n\n"
            f"QUESTION: {question}"
        )

    def generate(self, question: str, contexts: List[str]) -> str:
        prompt = self._build_prompt(question, contexts)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        except APIError as err:
            raise HTTPException(
                status_code=502, detail=f"OpenAI API Error: {err.message}"
            ) from err

        return resp.choices[0].message.content  # type: ignore[attr-defined]

# src/adapters/generation/openai_chat.py
"""
OpenAI Chat adapter(API v1.x.x).
"""
from typing import List
from openai import OpenAI, APIError
from fastapi import HTTPException

from src.core.ports import GeneratorPort
from src.settings import settings

class OpenAIGenerator(GeneratorPort):
    def __init__(self):
        if not settings.openai_api_key:
            pass
        
        self.client = OpenAI(
            # Could be None if env var expected to work
            api_key=settings.openai_api_key 
        )

    def generate(self, question: str, contexts: List[str]) -> str:
        ctx_block = "\n".join(f"- {c}" for c in contexts)
        prompt_content = ( 
            "Answer using ONLY the context provided.\n\n"
            f"CONTEXT:\n{ctx_block}\n\n"
            f"QUESTION: {question}"
        )
        try:
            # API > 1.0.0
            completion = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt_content
                    }
                ],
                temperature=settings.openai_temperature,
                top_p=settings.openai_top_p,
                max_tokens=settings.openai_max_tokens,
            )
            # Response
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content.strip()
            else:
                # Fallback
                raise HTTPException(500, detail="OpenAI response malformed: No content found.")

        except APIError as err: 
            # err.status_code, err.message, etc.
            error_detail = f"OpenAI API Error: {err.message}"
            if hasattr(err, 'status_code') and err.status_code:
                 error_detail = f"OpenAI API Error (Status {err.status_code}): {err.message}"
            # 502 for gateway/upstream errors
            raise HTTPException(502, detail=error_detail) from err
        except Exception as e:
            raise HTTPException(500, detail=f"Unexpected error during OpenAI call: {str(e)}") from e
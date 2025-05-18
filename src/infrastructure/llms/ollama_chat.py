# src/adapters/generation/ollama_chat.py
import logging
from typing import List

import requests
from fastapi import HTTPException

from src.core.ports import GeneratorPort
from src.settings import (  # settings.ollama_base_url y settings.ollama_request_timeout exists
    settings,
)

logger = logging.getLogger(__name__)


class OllamaGenerator(GeneratorPort):
    def generate(self, question: str, contexts: List[str]) -> str:
        ctx_block = "\n".join(f"- {c}" for c in contexts)
        full_prompt = (
            "Based on the following context, please answer the question.\nIf the context does not provide an answer, say so.\n\n"
            "CONTEXT:\n"
            f"{ctx_block}\n\n"
            "QUESTION:\n"
            f"{question}"
        )

        payload = {
            "model": settings.ollama_model,
            "prompt": full_prompt,
            "stream": False,  # Ollama by default returns the full response if stream equals false
            # "options": {"temperature": 0.7} #  OPTIONAL
        }

        base_url = settings.ollama_base_url.rstrip("/")
        api_url = f"{base_url}/api/generate"

        try:
            response = requests.post(
                api_url, json=payload, timeout=settings.ollama_request_timeout
            )
            response.raise_for_status()  # HTTP codes 4xx/5xx

            response_data = response.json()

            # The endpoint /api/generate retuns a JSON where every line it's a JSON if stream = True (default), else only 1 json with full answer:
            # when stream=False:
            # {
            #   "model": "...", "created_at": "...", "response": "...", "done": true,
            #   "context": [...], "total_duration": ..., ...
            # }
            if "response" in response_data and isinstance(
                response_data["response"], str
            ):
                return response_data["response"].strip()
            else:
                # logger.warning(f"Ollama response malformed. Data: {response_data}")
                raise HTTPException(
                    500,
                    detail="Ollama response malformed: 'response' key missing or not a string.",
                )

        except requests.exceptions.Timeout as err:
            raise HTTPException(
                504,
                detail=f"Ollama request timed out after {settings.ollama_request_timeout}s: {api_url}",
            ) from err
        except requests.exceptions.ConnectionError as err:
            raise HTTPException(
                503, detail=f"Could not connect to Ollama server at {api_url}"
            ) from err
        except requests.exceptions.HTTPError as err:
            error_content = err.response.text if err.response is not None else str(err)
            status_code = err.response.status_code if err.response is not None else 500
            raise HTTPException(
                status_code, detail=f"Ollama API error: {error_content}"
            ) from err
        except requests.exceptions.JSONDecodeError as err:
            # logger.error(f"Failed to decode Ollama JSON response. Status: {response.status_code}, Content: {response.text}")
            raise HTTPException(
                500,
                detail=f"Failed to decode Ollama JSON response. Original error: {str(err)}",
            ) from err
        except Exception as e:
            # logger.exception("Unexpected error during Ollama call") # Log con traceback
            raise HTTPException(
                500, detail=f"Unexpected error during Ollama call: {str(e)}"
            ) from e

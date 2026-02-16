"""
Map typed runtime errors to HTTP transport errors.
"""

from __future__ import annotations

from fastapi import HTTPException

from local_rag_backend.core.errors import (
    LLMConfigurationError,
    LLMConnectionError,
    LLMProviderError,
    LLMResponseError,
    LLMTimeoutError,
)


def raise_http_for_runtime_error(exc: Exception) -> None:
    """Raise an HTTPException if this error has a transport-level mapping."""
    if isinstance(exc, LLMTimeoutError):
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    if isinstance(exc, LLMConnectionError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, LLMResponseError):
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if isinstance(exc, LLMConfigurationError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if isinstance(exc, LLMProviderError):
        raise HTTPException(status_code=502, detail=str(exc)) from exc


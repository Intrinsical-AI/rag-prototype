"""Map typed runtime errors to app-level transport-agnostic errors."""

from __future__ import annotations

from local_rag_backend.app.errors import (
    AppError,
    BadGatewayError,
    BadRequestError,
    GatewayTimeoutError,
    InternalServerError,
    ServiceUnavailableError,
)
from local_rag_backend.core.errors import (
    EmbeddingsBackendUnavailableError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMProviderError,
    LLMResponseError,
    LLMTimeoutError,
)


def map_runtime_error(exc: Exception) -> AppError | None:
    """Return mapped AppError for known runtime error types."""
    if isinstance(exc, EmbeddingsBackendUnavailableError):
        return BadRequestError(str(exc))
    if isinstance(exc, LLMTimeoutError):
        return GatewayTimeoutError(str(exc))
    if isinstance(exc, LLMConnectionError):
        return ServiceUnavailableError(str(exc))
    if isinstance(exc, LLMResponseError):
        return BadGatewayError(str(exc))
    if isinstance(exc, LLMConfigurationError):
        return InternalServerError(str(exc))
    if isinstance(exc, LLMProviderError):
        return BadGatewayError(str(exc))
    return None

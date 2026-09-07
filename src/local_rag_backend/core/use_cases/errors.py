"""Application-level typed errors (transport-agnostic)."""

from __future__ import annotations

from typing import Any

from local_rag_backend.core.errors import (
    EmbeddingsBackendUnavailableError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMProviderError,
    LLMResponseError,
    LLMTimeoutError,
    WriteLockTimeoutError,
)


class AppError(Exception):
    """Base class for app errors that can be rendered by HTTP handlers."""

    status_code = 500
    default_detail = "Internal server error."

    def __init__(self, detail: Any | None = None) -> None:
        self.detail = self.default_detail if detail is None else detail
        super().__init__(str(self.detail))


class BadRequestError(AppError):
    status_code = 400
    default_detail = "Bad request."


class UnauthorizedError(AppError):
    status_code = 401
    default_detail = "Unauthorized."


class NotFoundError(AppError):
    status_code = 404
    default_detail = "Not found."


class ConflictError(AppError):
    status_code = 409
    default_detail = "Conflict."


class PayloadTooLargeError(AppError):
    status_code = 413
    default_detail = "Payload too large."


class UnprocessableEntityError(AppError):
    status_code = 422
    default_detail = "Unprocessable entity."


class BadGatewayError(AppError):
    status_code = 502
    default_detail = "Bad gateway."


class ServiceUnavailableError(AppError):
    status_code = 503
    default_detail = "Service unavailable."


class IndexRebuildRequiredError(ServiceUnavailableError):
    default_detail = (
        "Canonical scope deletion did not complete; the vector index may have changed. "
        "Rebuild from SQL with `rag-rebuild-index` or POST /api/index/rebuild "
        "before retrying."
    )


class GatewayTimeoutError(AppError):
    status_code = 504
    default_detail = "Gateway timeout."


class InternalServerError(AppError):
    status_code = 500
    default_detail = "Internal server error."


def map_runtime_error(exc: Exception) -> AppError | None:
    """Return mapped AppError for known runtime error types."""
    if isinstance(exc, EmbeddingsBackendUnavailableError):
        return BadRequestError(str(exc))
    if isinstance(exc, WriteLockTimeoutError):
        return ServiceUnavailableError(str(exc))
    if isinstance(exc, LLMTimeoutError):
        return GatewayTimeoutError(str(exc))
    if isinstance(exc, LLMConnectionError):
        return ServiceUnavailableError(str(exc))
    if isinstance(exc, LLMResponseError):
        return BadGatewayError(str(exc))
    if isinstance(exc, LLMConfigurationError):
        return ServiceUnavailableError(str(exc))
    if isinstance(exc, LLMProviderError):
        return BadGatewayError(str(exc))
    return None

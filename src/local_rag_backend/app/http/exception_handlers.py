"""Centralized HTTP exception handlers for app/core typed errors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse

from local_rag_backend.app.error_mapping import map_runtime_error
from local_rag_backend.app.errors import AppError, InternalServerError
from local_rag_backend.core.errors import (
    EmbeddingsBackendUnavailableError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMProviderError,
    LLMResponseError,
    LLMTimeoutError,
)

if TYPE_CHECKING:
    from fastapi import FastAPI


async def handle_app_error(_request: Request, exc: Exception) -> JSONResponse:
    app_error = exc if isinstance(exc, AppError) else InternalServerError(str(exc))
    return JSONResponse(
        status_code=int(app_error.status_code), content={"detail": app_error.detail}
    )


async def handle_runtime_error(request: Request, exc: Exception) -> JSONResponse:
    mapped = map_runtime_error(exc)
    app_error = mapped if mapped is not None else InternalServerError(str(exc))
    return await handle_app_error(request, app_error)


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, handle_app_error)
    app.add_exception_handler(EmbeddingsBackendUnavailableError, handle_runtime_error)
    app.add_exception_handler(LLMTimeoutError, handle_runtime_error)
    app.add_exception_handler(LLMConnectionError, handle_runtime_error)
    app.add_exception_handler(LLMResponseError, handle_runtime_error)
    app.add_exception_handler(LLMConfigurationError, handle_runtime_error)
    app.add_exception_handler(LLMProviderError, handle_runtime_error)

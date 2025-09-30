"""
Intrinsical-AI RAG Prototype
Copyright (c) 2025 Intrinsical-AI

Module: HTTP Middleware
Purpose: Observability middleware for request metrics and monitoring.
         Provides Prometheus-compatible metrics when enabled.
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from typing import TYPE_CHECKING

from local_rag_backend.settings import settings

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import Request, Response
    from starlette.types import ASGIApp


class MetricsMiddleware(BaseHTTPMiddleware):
    """HTTP metrics middleware for Prometheus if available and enabled."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self.is_active = PROMETHEUS_AVAILABLE and settings.enable_monitoring
        if self.is_active:
            self.requests = Counter(
                "http_requests_total", "Total HTTP requests", ["method", "path", "status_code"]
            )
            self.latencies = Histogram(
                "http_request_duration_seconds", "Request latency", ["method", "path"]
            )

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not self.is_active:
            return await call_next(request)

        start_time = time.time()
        response = await call_next(request)
        latency = time.time() - start_time

        self.latencies.labels(method=request.method, path=request.url.path).observe(latency)
        self.requests.labels(
            method=request.method, path=request.url.path, status_code=response.status_code
        ).inc()

        return response


def get_metrics() -> tuple[str, str]:
    """Get Prometheus metrics in text format if monitoring is active."""
    if not (PROMETHEUS_AVAILABLE and settings.enable_monitoring):
        return "# Monitoring disabled or prometheus-client not installed\n", "text/plain"
    return generate_latest().decode("utf-8"), CONTENT_TYPE_LATEST

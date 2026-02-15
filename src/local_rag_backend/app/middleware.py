# src/app/middleware.py
"""
Middleware for observability and monitoring.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Provide small stubs so tests can monkeypatch these symbols even when the optional
    # dependency isn't installed. The middleware remains inactive unless explicitly enabled.
    CONTENT_TYPE_LATEST = "text/plain"

    class _NoopMetric:  # pragma: no cover
        def __init__(self, *_a: object, **_k: object) -> None:
            return None

        def labels(self, **_kwargs: object) -> _NoopMetric:
            return self

        def inc(self, *_a: object, **_k: object) -> None:
            return None

        def observe(self, *_a: object, **_k: object) -> None:
            return None

    Counter = Histogram = _NoopMetric

    def generate_latest() -> bytes:  # pragma: no cover
        return b""


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

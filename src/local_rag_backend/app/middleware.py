# src/app/middleware.py
"""
Middleware for observability and monitoring.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from starlette.middleware.base import BaseHTTPMiddleware

from local_rag_backend.settings import settings

# prometheus-client is optional. Keep stable module-level symbols so tests can
# monkeypatch them even when the optional dependency isn't installed.
PROMETHEUS_AVAILABLE: bool
CONTENT_TYPE_LATEST: str
Counter: Any
Histogram: Any
generate_latest: Any


class _NoopMetric:  # pragma: no cover
    def __init__(self, *_a: object, **_k: object) -> None:
        return None

    def labels(self, **_kwargs: object) -> _NoopMetric:
        return self

    def inc(self, *_a: object, **_k: object) -> None:
        return None

    def observe(self, *_a: object, **_k: object) -> None:
        return None


def _noop_counter(*_args: Any, **_kwargs: Any) -> _NoopMetric:  # pragma: no cover
    return _NoopMetric()


def _noop_histogram(*_args: Any, **_kwargs: Any) -> _NoopMetric:  # pragma: no cover
    return _NoopMetric()


def _noop_generate_latest(*_args: Any, **_kwargs: Any) -> bytes:  # pragma: no cover
    return b""


try:  # pragma: no cover
    from prometheus_client import (
        CONTENT_TYPE_LATEST as _CONTENT_TYPE_LATEST,
        Counter as _PromCounter,
        Histogram as _PromHistogram,
        generate_latest as _prom_generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
    CONTENT_TYPE_LATEST = _CONTENT_TYPE_LATEST
    Counter = _PromCounter
    Histogram = _PromHistogram
    generate_latest = _prom_generate_latest
except ImportError:  # pragma: no cover
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"
    Counter = _noop_counter
    Histogram = _noop_histogram
    generate_latest = _noop_generate_latest

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

    @staticmethod
    def _path_label(request: Request, response_status: int) -> str:
        """
        Derive a low-cardinality label for request paths.

        High-cardinality labels (e.g., `/assets/<hash>.js` or random 404 paths) can lead to
        unbounded time-series growth and memory DoS in `prometheus_client`.
        """
        route = request.scope.get("route")
        route_path = getattr(route, "path", None)
        if isinstance(route_path, str) and route_path:
            return route_path

        # Unmatched routes (usually 404) should not create a new series per random path.
        if response_status == 404:
            return "<unmatched>"

        # Best-effort fallback. Keep this as stable as possible.
        raw_path = request.url.path
        if raw_path.startswith("/assets/"):
            return "/assets/*"
        return raw_path

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not self.is_active:
            return await call_next(request)

        start_time = time.time()
        response = await call_next(request)
        latency = time.time() - start_time

        path_label = self._path_label(request, response.status_code)
        self.latencies.labels(method=request.method, path=path_label).observe(latency)
        self.requests.labels(
            method=request.method, path=path_label, status_code=response.status_code
        ).inc()

        return response


def get_metrics() -> tuple[str, str]:
    """Get Prometheus metrics in text format if monitoring is active."""
    if not (PROMETHEUS_AVAILABLE and settings.enable_monitoring):
        return "# Monitoring disabled or prometheus-client not installed\n", "text/plain"
    return generate_latest().decode("utf-8"), CONTENT_TYPE_LATEST

"""
Middleware for observability and monitoring.
"""

import time
from collections.abc import Callable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from local_rag_backend.settings import settings


class MetricsMiddleware(BaseHTTPMiddleware):
    """HTTP metrics middleware for Prometheus monitoring."""

    def __init__(self, app: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(app, *args, **kwargs)

        if PROMETHEUS_AVAILABLE and settings.enable_monitoring:
            # HTTP request counter
            self.http_requests_total = Counter(
                "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status_code"]
            )

            # HTTP request duration histogram
            self.http_request_duration_seconds = Histogram(
                "http_request_duration_seconds",
                "HTTP request duration in seconds",
                ["method", "endpoint"],
            )
        else:
            self.http_requests_total = None
            self.http_request_duration_seconds = None

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        if not (PROMETHEUS_AVAILABLE and settings.enable_monitoring):
            return await call_next(request)

        # Record start time
        start_time = time.time()

        # Get endpoint pattern (remove query params)
        endpoint = request.url.path
        method = request.method

        # Process request
        response = await call_next(request)

        # Record metrics
        duration = time.time() - start_time
        status_code = str(response.status_code)

        if self.http_requests_total:
            self.http_requests_total.labels(
                method=method, endpoint=endpoint, status_code=status_code
            ).inc()

        if self.http_request_duration_seconds:
            self.http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(
                duration
            )

        return response


def get_metrics() -> tuple[str, str]:
    """
    Get Prometheus metrics in text format.
    Returns:
        tuple: (metrics_content, content_type)
    """
    if not (PROMETHEUS_AVAILABLE and settings.enable_monitoring):
        return "# Monitoring not enabled or prometheus-client not installed\n", "text/plain"

    return generate_latest().decode("utf-8"), CONTENT_TYPE_LATEST

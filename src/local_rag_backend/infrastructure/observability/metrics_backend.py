"""Prometheus backend primitives with a no-op fallback."""

from __future__ import annotations

from typing import Any

PROMETHEUS_AVAILABLE: bool
CONTENT_TYPE_LATEST: str
Counter: Any
Histogram: Any
Gauge: Any
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

    def set(self, *_a: object, **_k: object) -> None:
        return None


def _noop_counter(*_args: Any, **_kwargs: Any) -> _NoopMetric:  # pragma: no cover
    return _NoopMetric()


def _noop_histogram(*_args: Any, **_kwargs: Any) -> _NoopMetric:  # pragma: no cover
    return _NoopMetric()


def _noop_gauge(*_args: Any, **_kwargs: Any) -> _NoopMetric:  # pragma: no cover
    return _NoopMetric()


def _noop_generate_latest(*_args: Any, **_kwargs: Any) -> bytes:  # pragma: no cover
    return b""


try:  # pragma: no cover
    from prometheus_client import (
        CONTENT_TYPE_LATEST as _CONTENT_TYPE_LATEST,
        Counter as _PromCounter,
        Gauge as _PromGauge,
        Histogram as _PromHistogram,
        generate_latest as _prom_generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
    CONTENT_TYPE_LATEST = _CONTENT_TYPE_LATEST
    Counter = _PromCounter
    Histogram = _PromHistogram
    Gauge = _PromGauge
    generate_latest = _prom_generate_latest
except ImportError:  # pragma: no cover
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"
    Counter = _noop_counter
    Histogram = _noop_histogram
    Gauge = _noop_gauge
    generate_latest = _noop_generate_latest


__all__ = [
    "CONTENT_TYPE_LATEST",
    "PROMETHEUS_AVAILABLE",
    "Counter",
    "Gauge",
    "Histogram",
    "generate_latest",
]

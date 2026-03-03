"""Telemetry interfaces and default implementation for app metrics/logging."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Literal, Protocol

from local_rag_backend.infrastructure.observability import metrics_backend as mb

if TYPE_CHECKING:
    from local_rag_backend.infrastructure.concurrency.blocking import BlockingTaskType


BlockingStatus = Literal["ok", "error", "cancelled", "rejected"]

logger = logging.getLogger(__name__)


class TelemetrySink(Protocol):
    def log_event(self, event: str, **fields: Any) -> None: ...

    def observe_query(
        self,
        *,
        retrieval_mode: str,
        reranker_enabled: bool,
        ok: bool,
        duration_s: float,
    ) -> None: ...

    def observe_ingest(self, *, source: str, ok: bool, inserted: int) -> None: ...

    def observe_blocking_queue(
        self,
        *,
        task_type: BlockingTaskType,
        pending: int,
        capacity: int,
    ) -> None: ...

    def observe_blocking_queue_wait(
        self, *, task_type: BlockingTaskType, wait_s: float
    ) -> None: ...

    def observe_blocking_run(
        self,
        *,
        task_type: BlockingTaskType,
        status: BlockingStatus,
        duration_s: float,
    ) -> None: ...


class PrometheusTelemetry:
    """Default telemetry sink backed by Prometheus metrics and stdlib logging."""

    def __init__(self) -> None:
        self._queries_total = mb.Counter(
            "rag_queries_total",
            "Total RAG queries processed (ask endpoint).",
            ["retrieval_mode", "reranker_enabled", "status"],
        )
        self._query_latency = mb.Histogram(
            "rag_query_duration_seconds",
            "RAG query latency in seconds (ask endpoint).",
            ["retrieval_mode", "reranker_enabled"],
        )
        self._ingest_requests_total = mb.Counter(
            "rag_ingest_requests_total",
            "Total ingest requests processed (docs endpoints).",
            ["source", "status"],
        )
        self._ingest_docs_total = mb.Counter(
            "rag_ingest_docs_total",
            "Total documents/chunks ingested (docs endpoints).",
            ["source"],
        )

        # Blocking/offload health metrics (queue pressure + outcomes).
        self._blocking_pending = mb.Gauge(
            "rag_blocking_pending_tasks",
            "Current number of pending blocking tasks (queued+running).",
            ["task_type"],
        )
        self._blocking_capacity = mb.Gauge(
            "rag_blocking_capacity_tasks",
            "Configured pending capacity for blocking task pool.",
            ["task_type"],
        )
        self._blocking_saturation = mb.Gauge(
            "rag_blocking_saturation_ratio",
            "Pending/capacity ratio for blocking pool saturation tracking.",
            ["task_type"],
        )
        self._blocking_queue_wait = mb.Histogram(
            "rag_blocking_queue_wait_seconds",
            "Time spent waiting in blocking queue before execution starts.",
            ["task_type"],
        )
        self._blocking_runs_total = mb.Counter(
            "rag_blocking_runs_total",
            "Total offloaded blocking calls by status.",
            ["task_type", "status"],
        )
        self._blocking_run_duration = mb.Histogram(
            "rag_blocking_run_duration_seconds",
            "Execution duration for offloaded blocking calls.",
            ["task_type", "status"],
        )

    def log_event(self, event: str, **fields: Any) -> None:
        payload = {"event": event, **fields}
        try:
            logger.info(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        except Exception:  # pragma: no cover
            logger.info("%s %s", event, fields)

    def observe_query(
        self,
        *,
        retrieval_mode: str,
        reranker_enabled: bool,
        ok: bool,
        duration_s: float,
    ) -> None:
        mode = str(retrieval_mode)
        rerank = "1" if bool(reranker_enabled) else "0"
        self._query_latency.labels(retrieval_mode=mode, reranker_enabled=rerank).observe(
            float(duration_s)
        )
        self._queries_total.labels(
            retrieval_mode=mode,
            reranker_enabled=rerank,
            status="ok" if ok else "error",
        ).inc()

    def observe_ingest(self, *, source: str, ok: bool, inserted: int) -> None:
        src = str(source)
        self._ingest_requests_total.labels(source=src, status="ok" if ok else "error").inc()
        if inserted > 0:
            self._ingest_docs_total.labels(source=src).inc(float(inserted))

    def observe_blocking_queue(
        self,
        *,
        task_type: BlockingTaskType,
        pending: int,
        capacity: int,
    ) -> None:
        task = str(task_type)
        pending_value = max(0, int(pending))
        capacity_value = max(1, int(capacity))
        saturation = float(pending_value) / float(capacity_value)
        self._blocking_pending.labels(task_type=task).set(float(pending_value))
        self._blocking_capacity.labels(task_type=task).set(float(capacity_value))
        self._blocking_saturation.labels(task_type=task).set(saturation)

    def observe_blocking_queue_wait(self, *, task_type: BlockingTaskType, wait_s: float) -> None:
        self._blocking_queue_wait.labels(task_type=str(task_type)).observe(max(0.0, float(wait_s)))

    def observe_blocking_run(
        self,
        *,
        task_type: BlockingTaskType,
        status: BlockingStatus,
        duration_s: float,
    ) -> None:
        task = str(task_type)
        normalized_status = str(status)
        self._blocking_runs_total.labels(task_type=task, status=normalized_status).inc()
        self._blocking_run_duration.labels(task_type=task, status=normalized_status).observe(
            max(0.0, float(duration_s))
        )


_TELEMETRY: TelemetrySink = PrometheusTelemetry()


def get_telemetry() -> TelemetrySink:
    return _TELEMETRY


def set_telemetry(telemetry: TelemetrySink) -> None:
    global _TELEMETRY
    _TELEMETRY = telemetry


def reset_telemetry() -> None:
    global _TELEMETRY
    _TELEMETRY = PrometheusTelemetry()


__all__ = [
    "BlockingStatus",
    "PrometheusTelemetry",
    "TelemetrySink",
    "get_telemetry",
    "reset_telemetry",
    "set_telemetry",
]

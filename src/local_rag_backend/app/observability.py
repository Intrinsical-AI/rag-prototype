"""
Minimal observability primitives: structured-ish logs and domain metrics.

Goals:
- Keep dependencies optional (prometheus-client may be missing).
- Avoid high-cardinality labels.
- Avoid logging raw user content by default (privacy).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from local_rag_backend.app import middleware as mw
from local_rag_backend.settings import settings

logger = logging.getLogger(__name__)


def _sha256_short(text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return h[:12]


def log_event(event: str, **fields: Any) -> None:
    """
    Log a single event as a JSON string.

    This intentionally logs a JSON payload rather than relying on optional structlog,
    keeping behavior stable across installs.
    """
    payload = {"event": event, **fields}
    try:
        logger.info(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    except Exception:  # pragma: no cover
        logger.info("%s %s", event, fields)


# --- Domain metrics (optional Prometheus) --- #

_queries_total = mw.Counter(
    "rag_queries_total",
    "Total RAG queries processed (ask endpoint).",
    ["retrieval_mode", "reranker_enabled", "status"],
)
_query_latency = mw.Histogram(
    "rag_query_duration_seconds",
    "RAG query latency in seconds (ask endpoint).",
    ["retrieval_mode", "reranker_enabled"],
)
_ingest_requests_total = mw.Counter(
    "rag_ingest_requests_total",
    "Total ingest requests processed (docs endpoints).",
    ["source", "status"],
)
_ingest_docs_total = mw.Counter(
    "rag_ingest_docs_total",
    "Total documents/chunks ingested (docs endpoints).",
    ["source"],
)


def _labels_retrieval() -> tuple[str, str]:
    return str(settings.retrieval_mode), "1" if bool(settings.enable_reranker) else "0"


class Timer:
    def __init__(self) -> None:
        self._start = time.monotonic()

    def seconds(self) -> float:
        return float(time.monotonic() - self._start)


def observe_query(*, ok: bool, duration_s: float) -> None:
    mode, rerank = _labels_retrieval()
    _query_latency.labels(retrieval_mode=mode, reranker_enabled=rerank).observe(duration_s)
    _queries_total.labels(
        retrieval_mode=mode, reranker_enabled=rerank, status="ok" if ok else "error"
    ).inc()


def observe_ingest(*, source: str, ok: bool, inserted: int) -> None:
    src = str(source)
    _ingest_requests_total.labels(source=src, status="ok" if ok else "error").inc()
    if inserted > 0:
        _ingest_docs_total.labels(source=src).inc(float(inserted))


def fingerprint_question(question: str) -> dict[str, Any]:
    q = str(question or "")
    return {"q_sha256": _sha256_short(q), "q_len": len(q)}

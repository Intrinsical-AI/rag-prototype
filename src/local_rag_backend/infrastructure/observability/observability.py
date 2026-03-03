"""
Minimal observability primitives: structured-ish logs and domain metrics.

Goals:
- Keep dependencies optional (prometheus-client may be missing).
- Avoid high-cardinality labels.
- Avoid logging raw user content by default (privacy).
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from local_rag_backend.infrastructure.observability.telemetry import get_telemetry


def _sha256_short(text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return h[:12]


def log_event(event: str, **fields: Any) -> None:
    get_telemetry().log_event(event, **fields)


class Timer:
    def __init__(self) -> None:
        self._start = time.monotonic()

    def seconds(self) -> float:
        return float(time.monotonic() - self._start)


def observe_query(
    *,
    retrieval_mode: str,
    reranker_enabled: bool,
    ok: bool,
    duration_s: float,
) -> None:
    get_telemetry().observe_query(
        retrieval_mode=retrieval_mode,
        reranker_enabled=reranker_enabled,
        ok=ok,
        duration_s=duration_s,
    )


def observe_ingest(*, source: str, ok: bool, inserted: int) -> None:
    get_telemetry().observe_ingest(source=source, ok=ok, inserted=inserted)


def fingerprint_question(question: str) -> dict[str, Any]:
    q = str(question or "")
    return {"q_sha256": _sha256_short(q), "q_len": len(q)}

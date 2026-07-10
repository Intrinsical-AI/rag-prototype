"""Process-local perf metrics with optional JSON emission on exit."""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

from local_rag_backend.settings import settings

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {
    "embedding_cache": {
        "lookups": 0,
        "hits": 0,
        "misses": 0,
        "embedded_vectors": 0,
        "stored_vectors": 0,
        "lookup_seconds": 0.0,
        "embed_seconds": 0.0,
        "store_seconds": 0.0,
        "errors": 0,
    }
}


def reset_perf_metrics() -> None:
    with _LOCK:
        _STATE["embedding_cache"] = {
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "embedded_vectors": 0,
            "stored_vectors": 0,
            "lookup_seconds": 0.0,
            "embed_seconds": 0.0,
            "store_seconds": 0.0,
            "errors": 0,
        }


def record_embedding_cache_lookup(*, hits: int, misses: int, seconds: float) -> None:
    with _LOCK:
        stats = _STATE["embedding_cache"]
        stats["lookups"] += int(hits) + int(misses)
        stats["hits"] += int(hits)
        stats["misses"] += int(misses)
        stats["lookup_seconds"] += float(seconds)


def record_embedding_cache_embed(*, count: int, seconds: float) -> None:
    with _LOCK:
        stats = _STATE["embedding_cache"]
        stats["embedded_vectors"] += int(count)
        stats["embed_seconds"] += float(seconds)


def record_embedding_cache_store(*, count: int, seconds: float) -> None:
    with _LOCK:
        stats = _STATE["embedding_cache"]
        stats["stored_vectors"] += int(count)
        stats["store_seconds"] += float(seconds)


def record_embedding_cache_error() -> None:
    with _LOCK:
        _STATE["embedding_cache"]["errors"] += 1


def snapshot_perf_metrics() -> dict[str, Any]:
    with _LOCK:
        payload = deepcopy(_STATE)
    cache = payload["embedding_cache"]
    lookups = int(cache["lookups"])
    hits = int(cache["hits"])
    embedded = int(cache["embedded_vectors"])
    embed_seconds = float(cache["embed_seconds"])
    cache["hit_rate"] = round((hits / lookups), 6) if lookups else 0.0
    cache["avg_embed_seconds"] = round((embed_seconds / embedded), 6) if embedded else 0.0
    cache["estimated_saved_embed_seconds"] = (
        round(
            (embed_seconds / embedded) * hits,
            6,
        )
        if embedded and hits
        else 0.0
    )
    return payload


def _merge_process_metrics(path: Path, payload: dict[str, Any]) -> None:
    existing: dict[str, Any] = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    processes = list(existing.get("processes") or [])
    processes.append(payload)

    summary = {
        "embedding_cache": {
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "embedded_vectors": 0,
            "stored_vectors": 0,
            "lookup_seconds": 0.0,
            "embed_seconds": 0.0,
            "store_seconds": 0.0,
            "errors": 0,
            "estimated_saved_embed_seconds": 0.0,
        }
    }
    for process in processes:
        stats = (process.get("metrics") or {}).get("embedding_cache") or {}
        aggregate = summary["embedding_cache"]
        for key in (
            "lookups",
            "hits",
            "misses",
            "embedded_vectors",
            "stored_vectors",
            "errors",
        ):
            aggregate[key] += int(stats.get(key, 0) or 0)
        for key in (
            "lookup_seconds",
            "embed_seconds",
            "store_seconds",
            "estimated_saved_embed_seconds",
        ):
            aggregate[key] += float(stats.get(key, 0.0) or 0.0)
    lookups = int(summary["embedding_cache"]["lookups"])
    hits = int(summary["embedding_cache"]["hits"])
    summary["embedding_cache"]["hit_rate"] = round((hits / lookups), 6) if lookups else 0.0

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"processes": processes, "summary": summary}, indent=2) + "\n",
        encoding="utf-8",
    )


def dump_perf_metrics_if_configured() -> None:
    output = str(getattr(settings, "perf_metrics_out_path", "") or "").strip()
    if not output:
        return
    payload = {
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "metrics": snapshot_perf_metrics(),
    }
    try:
        _merge_process_metrics(Path(output), payload)
    except Exception:
        return


atexit.register(dump_perf_metrics_if_configured)

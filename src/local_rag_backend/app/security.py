"""
Minimal API-key auth for deployments where the service is reachable beyond localhost.

This project is intentionally lightweight (demo/prototype). The goal is not to provide
full authN/Z, but to prevent accidental exposure of costly endpoints (LLM proxy, ingestion)
when binding to 0.0.0.0 or running in shared environments.
"""

from __future__ import annotations

import hmac

from fastapi import HTTPException, Request

from local_rag_backend.settings import settings

API_KEY_HEADER = "X-API-Key"


async def require_api_key(request: Request) -> None:
    """
    If `settings.api_key` is set, require clients to send the same value in `X-API-Key`.

    If no API key is configured, this dependency is a no-op (dev-friendly default).
    """
    expected = settings.api_key
    if not expected:
        return

    provided = request.headers.get(API_KEY_HEADER) or ""
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")

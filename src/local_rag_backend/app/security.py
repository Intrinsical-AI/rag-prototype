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
_LOCALHOST_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _normalize_host_token(raw: str) -> str:
    token = str(raw or "").strip().lower()
    if not token:
        return ""
    if token.startswith("[") and "]" in token:
        return token[1 : token.index("]")]
    if token.count(":") == 1:
        host_part, port_part = token.rsplit(":", 1)
        if port_part.isdigit():
            return host_part
    return token


def _request_hosts(request: Request) -> list[str]:
    hosts: list[str] = []
    client_host = _normalize_host_token(request.client.host if request.client else "")
    if client_host:
        hosts.append(client_host)

    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        hosts.extend(_normalize_host_token(part) for part in xff.split(",") if part.strip())
    return hosts


def enforce_safe_bind_config() -> None:
    """
    Refuse to start with a public bind without an API key.

    Threat model: this is a "local-first" service, but users sometimes run it in Docker
    with `-p 8000:8000` or set `APP_HOST=0.0.0.0`. Without auth, endpoints can ingest
    arbitrary docs and proxy paid LLM calls (OpenAI/OpenRouter), causing data/cost risk.
    """
    if not getattr(settings, "public_bind_requires_api_key", True):
        return
    if settings.api_key:
        return

    host = (settings.app_host or "").strip().lower()
    localhost_hosts = {"127.0.0.1", "localhost", "::1"}
    if host in localhost_hosts:
        return

    raise RuntimeError(
        "Refusing to start without API key when binding to a non-localhost address. "
        "Set API_KEY (recommended) or set PUBLIC_BIND_REQUIRES_API_KEY=false to override."
    )


async def require_api_key(request: Request) -> None:
    """
    If `settings.api_key` is set, require clients to send the same value in `X-API-Key`.

    If no API key is configured, this dependency is a no-op (dev-friendly default).
    """
    expected = settings.api_key
    if not expected:
        if not getattr(settings, "public_bind_requires_api_key", True):
            return
        hosts = _request_hosts(request)
        if not hosts or any(host not in _LOCALHOST_HOSTS for host in hosts):
            raise HTTPException(
                status_code=401,
                detail=(
                    "Unauthorized: API key is required for non-local requests. "
                    "Set API_KEY or disable PUBLIC_BIND_REQUIRES_API_KEY explicitly."
                ),
            )
        return

    provided = request.headers.get(API_KEY_HEADER) or ""
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")

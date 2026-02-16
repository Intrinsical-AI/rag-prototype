"""
OpenAI client construction helpers with compatibility fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai import OpenAI

if TYPE_CHECKING:
    from collections.abc import Callable


def create_openai_client(
    *,
    api_key: str | None,
    base_url: str | None = None,
    default_headers: dict[str, str] | None = None,
    timeout: int | float | None = None,
    client_factory: Callable[..., Any] = OpenAI,
) -> Any:
    """
    Build an OpenAI-compatible client and fallback when `timeout` is unsupported.
    """
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        kwargs["base_url"] = base_url
    if default_headers is not None:
        kwargs["default_headers"] = default_headers
    if timeout is not None:
        kwargs["timeout"] = timeout
    try:
        return client_factory(**kwargs)
    except TypeError as e:
        if "timeout" not in str(e):
            raise
        kwargs.pop("timeout", None)
        return client_factory(**kwargs)

from __future__ import annotations

import pytest

from local_rag_backend.infrastructure.llms.openai_chat import create_openai_client


def test_create_openai_client_omits_none_optional_kwargs() -> None:
    captured: dict[str, object] = {}

    def _factory(**kwargs):
        captured.update(kwargs)
        return "ok"

    out = create_openai_client(
        api_key="k",
        base_url=None,
        default_headers=None,
        timeout=None,
        client_factory=_factory,
    )
    assert out == "ok"
    assert captured == {"api_key": "k"}


def test_create_openai_client_falls_back_when_timeout_is_unsupported() -> None:
    calls: list[dict[str, object]] = []

    def _factory(**kwargs):
        calls.append(dict(kwargs))
        if "timeout" in kwargs:
            raise TypeError("__init__() got an unexpected keyword argument 'timeout'")
        return "ok"

    out = create_openai_client(
        api_key="k",
        timeout=17,
        client_factory=_factory,
    )
    assert out == "ok"
    assert calls == [{"api_key": "k", "timeout": 17}, {"api_key": "k"}]


def test_create_openai_client_reraises_unrelated_typeerror() -> None:
    def _factory(**_kwargs):
        raise TypeError("unexpected keyword argument 'base_url'")

    with pytest.raises(TypeError, match="base_url"):
        create_openai_client(
            api_key="k",
            base_url="https://x.invalid",
            client_factory=_factory,
        )

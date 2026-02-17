from __future__ import annotations

import pytest
from fastapi import HTTPException

from local_rag_backend.app.error_mapping import raise_http_for_runtime_error
from local_rag_backend.core.errors import (
    EmbeddingsBackendUnavailableError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMProviderError,
    LLMResponseError,
    LLMTimeoutError,
)


@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (EmbeddingsBackendUnavailableError("embeddings"), 400),
        (LLMTimeoutError("timeout"), 504),
        (LLMConnectionError("connection"), 503),
        (LLMResponseError("response"), 502),
        (LLMConfigurationError("config"), 500),
        (LLMProviderError("provider"), 502),
    ],
)
def test_raise_http_for_runtime_error_maps_typed_errors(
    exc: Exception, expected_status: int
) -> None:
    with pytest.raises(HTTPException) as raised:
        raise_http_for_runtime_error(exc)

    assert raised.value.status_code == expected_status
    assert raised.value.detail == str(exc)


def test_raise_http_for_runtime_error_ignores_unknown_error() -> None:
    raise_http_for_runtime_error(RuntimeError("unmapped"))

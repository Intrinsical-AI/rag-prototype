from __future__ import annotations

import pytest

from local_rag_backend.core.errors import (
    EmbeddingsBackendUnavailableError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMProviderError,
    LLMResponseError,
    LLMTimeoutError,
)
from local_rag_backend.core.use_cases.errors import AppError, map_runtime_error


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
def test_map_runtime_error_maps_typed_errors(exc: Exception, expected_status: int) -> None:
    mapped = map_runtime_error(exc)
    assert isinstance(mapped, AppError)
    assert mapped.status_code == expected_status
    assert str(mapped.detail) == str(exc)


def test_map_runtime_error_returns_none_for_unknown_error() -> None:
    assert map_runtime_error(RuntimeError("unmapped")) is None

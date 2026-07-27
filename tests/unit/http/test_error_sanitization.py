from __future__ import annotations

import pytest
from starlette.requests import Request

from local_rag_backend.core.use_cases.errors import (
    BadGatewayError,
    GatewayTimeoutError,
    InternalServerError,
    ServiceUnavailableError,
)
from local_rag_backend.http.exception_handlers import handle_app_error
from local_rag_backend.settings import settings


async def test_handle_app_error_hides_500_details_when_not_debug(monkeypatch):
    monkeypatch.setattr(settings, "debug", False, raising=False)
    request = Request({"type": "http", "method": "GET", "path": "/", "headers": []})
    response = await handle_app_error(request, InternalServerError("secret-backend-detail"))

    assert response.status_code == 500
    assert b"secret-backend-detail" not in response.body
    assert b"Internal server error." in response.body


async def test_handle_app_error_keeps_500_details_in_debug(monkeypatch):
    monkeypatch.setattr(settings, "debug", True, raising=False)
    request = Request({"type": "http", "method": "GET", "path": "/", "headers": []})
    response = await handle_app_error(request, InternalServerError("debug-detail"))

    assert response.status_code == 500
    assert b"debug-detail" in response.body


@pytest.mark.parametrize(
    ("error", "status_code", "public_detail"),
    [
        (BadGatewayError("secret-upstream"), 502, b"Bad gateway."),
        (ServiceUnavailableError("secret-service"), 503, b"Service unavailable."),
        (GatewayTimeoutError("secret-timeout"), 504, b"Gateway timeout."),
    ],
)
async def test_handle_app_error_hides_all_production_5xx_details(
    monkeypatch, error, status_code, public_detail
):
    monkeypatch.setattr(settings, "debug", False, raising=False)
    request = Request({"type": "http", "method": "GET", "path": "/", "headers": []})

    response = await handle_app_error(request, error)

    assert response.status_code == status_code
    assert b"secret-" not in response.body
    assert public_detail in response.body

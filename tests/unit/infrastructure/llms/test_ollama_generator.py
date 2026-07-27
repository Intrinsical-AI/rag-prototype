# tests/unit/infrastructure/llms/test_ollama_generator.py

import httpx
import pytest

from local_rag_backend.core.errors import (
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
)
from local_rag_backend.infrastructure.llms.ollama_chat import OllamaGenerator


# ---------------- helpers -------------------------------------------------- #
class _RespOK:
    status_code = 200

    def json(self):
        return {"response": "answer"}

    def raise_for_status(self):
        pass


class _RespNoField(_RespOK):
    def json(self):
        return {"foo": "bar"}


# ---------------- tests ---------------------------------------------------- #
def test_generate_ok(monkeypatch):
    monkeypatch.setattr(
        "local_rag_backend.infrastructure.llms.ollama_chat.httpx.post", lambda *a, **k: _RespOK()
    )
    gen = OllamaGenerator()
    out = gen.generate("q", ["ctx1"])
    assert out == "answer"


def test_generate_missing_response(monkeypatch):
    monkeypatch.setattr(
        "local_rag_backend.infrastructure.llms.ollama_chat.httpx.post",
        lambda *a, **k: _RespNoField(),
    )
    gen = OllamaGenerator()
    with pytest.raises(LLMResponseError) as exc:
        gen.generate("q", ["ctx"])
    assert "response malformed" in str(exc.value).lower()


def test_generate_timeout(monkeypatch):
    def _timeout(*_, **__):
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr("local_rag_backend.infrastructure.llms.ollama_chat.httpx.post", _timeout)
    gen = OllamaGenerator()
    with pytest.raises(LLMTimeoutError) as exc:
        gen.generate("q", ["ctx"])
    assert str(exc.value) == "Ollama request timed out"


def test_generate_connection_error_does_not_expose_uri(monkeypatch):
    request = httpx.Request("POST", "http://internal-host:11434/api/generate")

    def _connect(*_, **__):
        raise httpx.ConnectError("secret internal URI", request=request)

    monkeypatch.setattr("local_rag_backend.infrastructure.llms.ollama_chat.httpx.post", _connect)
    with pytest.raises(LLMConnectionError) as exc:
        OllamaGenerator().generate("q", ["ctx"])
    assert str(exc.value) == "Could not connect to Ollama"
    assert "internal-host" not in str(exc.value)


def test_generate_http_error_does_not_expose_response_body(monkeypatch):
    request = httpx.Request("POST", "http://internal-host:11434/api/generate")
    response = httpx.Response(500, request=request, text="secret upstream body")

    def _http_error(*_, **__):
        raise httpx.HTTPStatusError("secret status", request=request, response=response)

    monkeypatch.setattr("local_rag_backend.infrastructure.llms.ollama_chat.httpx.post", _http_error)
    with pytest.raises(LLMResponseError) as exc:
        OllamaGenerator().generate("q", ["ctx"])
    assert str(exc.value) == "Ollama returned an error response"
    assert "secret" not in str(exc.value)


def test_generate_request_error_does_not_expose_detail(monkeypatch):
    request = httpx.Request("POST", "http://internal-host:11434/api/generate")

    def _request_error(*_, **__):
        raise httpx.RequestError("secret request detail", request=request)

    monkeypatch.setattr(
        "local_rag_backend.infrastructure.llms.ollama_chat.httpx.post", _request_error
    )
    with pytest.raises(LLMResponseError) as exc:
        OllamaGenerator().generate("q", ["ctx"])
    assert str(exc.value) == "Ollama request failed"
    assert "secret" not in str(exc.value)


def test_generate_unexpected_error_does_not_expose_detail(monkeypatch):
    def _unexpected(*_, **__):
        raise RuntimeError("secret unexpected detail")

    monkeypatch.setattr("local_rag_backend.infrastructure.llms.ollama_chat.httpx.post", _unexpected)
    with pytest.raises(LLMResponseError) as exc:
        OllamaGenerator().generate("q", ["ctx"])
    assert str(exc.value) == "Unexpected error calling Ollama"
    assert "secret" not in str(exc.value)

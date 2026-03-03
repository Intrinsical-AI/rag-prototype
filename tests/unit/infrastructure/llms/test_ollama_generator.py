# tests/unit/infrastructure/llms/test_ollama_generator.py

import httpx
import pytest

from local_rag_backend.core.errors import LLMResponseError, LLMTimeoutError
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
    with pytest.raises(LLMTimeoutError):
        gen.generate("q", ["ctx"])

# tests/unit/infrastructure/llms/test_ollama_generator.py

import pytest
import requests
from fastapi import HTTPException

from src.infrastructure.llms.ollama_chat import OllamaGenerator


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
    monkeypatch.setattr("requests.post", lambda *a, **k: _RespOK())
    gen = OllamaGenerator()
    out = gen.generate("q", ["ctx1"])
    assert out == "answer"


def test_generate_missing_response(monkeypatch):
    monkeypatch.setattr("requests.post", lambda *a, **k: _RespNoField())
    gen = OllamaGenerator()
    with pytest.raises(HTTPException) as exc:
        gen.generate("q", ["ctx"])
    assert exc.value.status_code == 500


def test_generate_timeout(monkeypatch):
    def _timeout(*_, **__):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr("requests.post", _timeout)
    gen = OllamaGenerator()
    with pytest.raises(HTTPException) as exc:
        gen.generate("q", ["ctx"])
    assert exc.value.status_code == 504

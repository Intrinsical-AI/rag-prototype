# tests/unit/infrastructure/llms/test_openai_generator.py

import pytest
from fastapi import HTTPException

from src.infrastructure.llms.openai_chat import OpenAIGenerator


# --------------------------------------------------------------------------- #
def make_dummy_openai(should_raise=False):
    class DummyComp:
        def create(self, **_):
            if should_raise:
                from openai import APIError

                raise APIError("boom", request=None)  # <-- Aquí el cambio

            class DummyResp:
                choices = [
                    type("Msg", (), {"message": type("Cont", (), {"content": "OK"})()})
                ]

            return DummyResp()

    class DummyChat:
        completions = DummyComp()

    class DummyClient:
        chat = DummyChat()

    return lambda *a, **k: DummyClient()


# --------------------------------------------------------------------------- #
def test_build_prompt_private_method(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    gen = OpenAIGenerator()
    prompt = gen._build_prompt(
        question="¿Qué tal?",
        contexts=["Uno", "Dos"],
    )
    assert "CONTEXT" in prompt and "QUESTION" in prompt
    assert "- Uno" in prompt and "- Dos" in prompt


def test_generate_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "DUMMY")

    monkeypatch.setattr(
        "src.infrastructure.llms.openai_chat.OpenAI", make_dummy_openai()
    )
    gen = OpenAIGenerator()
    out = gen.generate("hola", ["ctx"])
    assert out == "OK"


def test_generate_api_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "DUMMY")

    monkeypatch.setattr(
        "src.infrastructure.llms.openai_chat.OpenAI",
        make_dummy_openai(should_raise=True),
    )
    gen = OpenAIGenerator()
    with pytest.raises(HTTPException) as exc:
        gen.generate("fallará", ["ctx"])
    assert exc.value.status_code == 502

# tests/unit/app/test_factory_and_settings.py
import importlib

import pytest

from src.app import factory
from src.settings import settings


# ---------- helpers -----------------------------------------------------------
def reload_factory():
    """Recarga el módulo factory para que re-evalue las condiciones de settings."""
    importlib.reload(factory)


class DummySqlDocumentStorage:
    def get_all_documents(self):
        # devuelvo 2 docs «falsos» para evitar código que falla con corpus vacío
        from src.core.domain.entities import Document

        return [Document(id=1, content="D1"), Document(id=2, content="D2")]


class DummySqlDocumentStorageV2:
    def get_all_documents(self):
        from src.core.domain.entities import Document

        return [Document(id=1, content="D1"), Document(id=2, content="D2")]


# ---------- tests get_generator() ---------------------------------------------
@pytest.mark.parametrize(
    "ollama_enabled,openai_key,expected_cls",
    [
        (True, None, "OllamaGenerator"),
        (False, "KEY", "OpenAIGenerator"),
    ],
)
def test_get_generator_branching(monkeypatch, ollama_enabled, openai_key, expected_cls):
    monkeypatch.setattr(settings, "ollama_enabled", ollama_enabled, raising=False)
    monkeypatch.setattr(settings, "openai_api_key", openai_key, raising=False)

    reload_factory()
    gen = factory.get_generator()
    assert gen.__class__.__name__ == expected_cls


def test_get_generator_no_llm_configured(monkeypatch):
    monkeypatch.setattr(settings, "ollama_enabled", False, raising=False)
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)

    reload_factory()
    with pytest.raises(RuntimeError, match="No LLM generator"):
        factory.get_generator()


@pytest.mark.parametrize(
    "mode,patched_class_name",
    [
        ("sparse", "SPARSE"),
        ("dense", "DENSE"),
        ("hybrid", "HYBRID"),
    ],
)
def test_get_retriever_selects_correct_class(monkeypatch, mode, patched_class_name):
    # Parcheamos sobre factory las clases concretas
    monkeypatch.setattr(settings, "retrieval_mode", mode, raising=False)
    monkeypatch.setattr(
        factory, "SqlDocumentStorage", lambda: DummySqlDocumentStorage()
    )
    monkeypatch.setattr(factory, "DenseFaissRetriever", lambda *a, **k: "DENSE")
    monkeypatch.setattr(factory, "SparseBM25Retriever", lambda *a, **k: "SPARSE")
    monkeypatch.setattr(factory, "HybridRetriever", lambda *a, **k: "HYBRID")

    # No hace falta recargar módulo si los parches van bien
    retriever = factory.get_retriever()
    assert retriever == {"sparse": "SPARSE", "dense": "DENSE", "hybrid": "HYBRID"}[mode]


def test_get_retriever_invalid_mode(monkeypatch):
    monkeypatch.setattr(
        factory, "SqlDocumentStorage", lambda: DummySqlDocumentStorage(), raising=True
    )
    monkeypatch.setattr(settings, "retrieval_mode", "unknown", raising=False)
    reload_factory()
    with pytest.raises(ValueError, match="Unsupported retrieval_mode"):
        factory.get_retriever()

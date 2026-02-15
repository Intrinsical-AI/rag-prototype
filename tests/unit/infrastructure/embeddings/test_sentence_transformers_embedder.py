# tests/unit/infrastructure/embeddings/test_sentence_transformers_embedder.py
import builtins

import pytest

from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
    SentenceTransformerEmbedder,
)


def test_sentence_transformers_missing_dependency_message_mentions_dense_st(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("forced for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError) as exc:
        SentenceTransformerEmbedder()
    assert "dense-st" in str(exc.value)

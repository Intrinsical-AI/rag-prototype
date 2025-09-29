import pytest

# Skip if sentence_transformers is not available in the environment
pytest.importorskip("sentence_transformers")

from local_rag_backend.infrastructure.embeddings.sentence_transformers import (
    SentenceTransformerEmbedder,
)


def test_embedder_returns_correct_shape_and_type():
    texts = ["Hola mundo", "¿Cómo estás?", "Este es un texto de prueba."]
    emb = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    vectors = emb.embed(texts)

    # Should return list of same length
    assert isinstance(vectors, list)
    assert len(vectors) == len(texts)

    # Each vector should be list/tuple of floats of the expected dimension
    dim = emb.dim
    for vec in vectors:
        assert hasattr(vec, "__len__")
        assert len(vec) == dim
        # All its elements should be floats
        assert all(isinstance(x, float) for x in vec)


def test_embed_empty_list_returns_empty():
    emb = SentenceTransformerEmbedder()
    out = emb.embed([])
    assert out == []

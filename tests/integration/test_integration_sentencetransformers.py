from src.infrastructure.embeddings.sentence_transformers import (
    SentenceTransformerEmbedder,
)


def test_embedder_returns_correct_shape_and_type():
    texts = ["Hola mundo", "¿Cómo estás?", "Este es un texto de prueba."]
    emb = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    vectors = emb.embed(texts)

    # Debe devolver lista de misma longitud
    assert isinstance(vectors, list)
    assert len(vectors) == len(texts)

    # Cada vector debe ser lista/tuple de floats de la dimensión esperada
    dim = emb.dim
    for vec in vectors:
        assert hasattr(vec, "__len__")
        assert len(vec) == dim
        # y todos sus elementos son floats
        assert all(isinstance(x, float) for x in vec)


def test_embed_empty_list_returns_empty():
    emb = SentenceTransformerEmbedder()
    out = emb.embed([])
    assert out == []

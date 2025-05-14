# tests/unit/adapters/embeddings/test_sentence_transformer_embedder.py
from unittest import mock
import numpy as np # Para comparar arrays si es necesario

from src.adapters.embeddings.sentence_transformers import SentenceTransformerEmbedder
# Asumimos que SentenceTransformerEmbedder implementa EmbedderPort
# from src.core.ports import EmbedderPort 

# No necesitamos una fixture para instanciar SentenceTransformerEmbedder
# porque su __init__ es el que carga el modelo, y eso es lo que queremos mockear
# o controlar.

# El path para mockear es donde 'SentenceTransformer' (la clase de la librería)
# es importada y usada por tu adaptador.
@mock.patch("src.adapters.embeddings.sentence_transformers.SentenceTransformer")
def test_sentence_transformer_embedder_happy_path(MockSentenceTransformer):
    # Arrange
    model_name_expected = "all-MiniLM-L6-v2" # El default en tu adaptador
    text_to_embed = "This is a test sentence for sentence-transformer."
    
    # Configurar el mock de la instancia de SentenceTransformer y su método encode
    mock_model_instance = MockSentenceTransformer.return_value # Lo que SentenceTransformer(model_name) devolvería
    
    # SentenceTransformer().encode() devuelve un numpy array.
    # La DIM de tu adaptador es 384 para 'all-MiniLM-L6-v2'.
    expected_embedding_vector_np = np.array([i * 0.01 for i in range(384)], dtype=np.float32)
    # Tu adaptador convierte esto a una lista de Python floats.
    expected_embedding_list = expected_embedding_vector_np.tolist()
    
    mock_model_instance.encode.return_value = [expected_embedding_vector_np] # encode espera una lista de textos y devuelve una lista de arrays

    # Act
    # Instanciar el embedder. Su __init__ llamará a SentenceTransformer(model_name_expected),
    # que ahora está mockeado por MockSentenceTransformer.
    embedder = SentenceTransformerEmbedder(model_name=model_name_expected)
    embedding_result = embedder.embed(text_to_embed)

    # Assert
    # 1. Verificar que SentenceTransformer fue instanciado con el nombre de modelo correcto
    MockSentenceTransformer.assert_called_once_with(model_name_expected)
    
    # 2. Verificar que el método encode del modelo mockeado fue llamado correctamente
    mock_model_instance.encode.assert_called_once_with([text_to_embed]) # encode toma una lista de strings
    
    # 3. Verificar que el embedding devuelto es el esperado
    assert embedding_result == expected_embedding_list
    assert isinstance(embedding_result, list)
    assert len(embedding_result) == 384 # Verificar dimensión
    assert all(isinstance(x, float) for x in embedding_result) # Verificar tipo de los elementos

    # 4. Verificar el atributo DIM del embedder
    assert embedder.DIM == 384


@mock.patch("src.adapters.embeddings.sentence_transformers.SentenceTransformer")
def test_sentence_transformer_embedder_custom_model_name(MockSentenceTransformer):
    # Arrange
    custom_model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    text_to_embed = "Otra frase de prueba."
    
    mock_model_instance = MockSentenceTransformer.return_value
    # Asumimos que este modelo también tiene DIM 384 para simplificar el mock,
    # aunque en realidad podría ser diferente. El test se enfoca en que el nombre del modelo se pase.
    # Si la DIM fuera diferente, el embedder.DIM debería reflejarlo (requeriría lógica en __init__ para actualizar DIM)
    # o el test debería mockear la DIM de forma diferente.
    # Por ahora, mantenemos la DIM de 384.
    expected_embedding_vector_np = np.array([i * 0.02 for i in range(384)], dtype=np.float32)
    expected_embedding_list = expected_embedding_vector_np.tolist()
    mock_model_instance.encode.return_value = [expected_embedding_vector_np]

    # Act
    embedder = SentenceTransformerEmbedder(model_name=custom_model_name) # Usar nombre custom
    embedding_result = embedder.embed(text_to_embed)

    # Assert
    MockSentenceTransformer.assert_called_once_with(custom_model_name) # Verificar nombre custom
    mock_model_instance.encode.assert_called_once_with([text_to_embed])
    assert embedding_result == expected_embedding_list
    # Nota: El atributo embedder.DIM se hardcodea a 384 en tu clase actualmente.
    # Si quisieras que fuera dinámico según el modelo, el __init__ necesitaría obtenerlo del modelo.
    # Para el test actual, el DIM seguirá siendo 384.
    assert embedder.DIM == 384 
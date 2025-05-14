# tests/unit/test_retrieval_dense_faiss.py
import pytest
import numpy as np
import faiss
import pickle
from pathlib import Path

from src.settings import settings as global_settings # Para monkeypatch
from src.adapters.retrieval.dense_faiss import DenseFaissRetriever
from src.core.ports import EmbedderPort # Para nuestro DummyEmbedder

# --- Dummy Embedder ---
class DummyEmbedder(EmbedderPort):
    DIM: int # Debe coincidir con la dimensión del índice FAISS de prueba

    def __init__(self, dim: int, predefined_embeddings: dict[str, list[float]] = None):
        self.DIM = dim
        # predefined_embeddings: mapea texto de query/doc a su vector
        self.predefined_embeddings = predefined_embeddings if predefined_embeddings else {}
        # Default embedding si no se encuentra en predefined_embeddings
        self.default_vector = [0.0] * self.DIM

    def embed(self, text: str) -> list[float]:
        return self.predefined_embeddings.get(text, self.default_vector)

# --- Fixture para crear y limpiar artefactos de FAISS para el test ---
@pytest.fixture
def faiss_test_artefacts(tmp_path: Path) -> tuple[Path, Path, dict[str, list[float]], int]:
    dim = 4 # Dimensión pequeña para el test
    num_docs = 3

    # Documentos y sus embeddings predefinidos (simplificados)
    # Estos embeddings están diseñados para que la "query_target_doc1" esté más cerca de "doc1_text"
    doc_embeddings = {
        "doc1_text": [1.0, 0.1, 0.2, 0.3], # Target
        "doc2_text": [0.2, 1.0, 0.3, 0.4],
        "doc3_text": [0.3, 0.2, 1.0, 0.5],
    }
    doc_ids_in_db = [101, 102, 103] # IDs que estarían en la BD

    query_embeddings = {
        "query_target_doc1": [0.9, 0.15, 0.25, 0.35], # Muy similar a doc1_text
        "query_target_doc2": [0.25, 0.9, 0.35, 0.45], # Muy similar a doc2_text
        "query_no_match":    [0.0, 0.0, 0.0, 0.0],   # Diferente a todos
    }

    all_predefined_embeddings = {**doc_embeddings, **query_embeddings}

    # Crear el índice FAISS
    index_path = tmp_path / "test_index.faiss"
    id_map_path = tmp_path / "test_id_map.pkl"

    # Convertir doc_embeddings a numpy array para FAISS
    vectors_np = np.array(list(doc_embeddings.values()), dtype="float32")
    
    # Usar IndexFlatL2 para similaridad de coseno (después de normalizar) o distancia L2
    # Para este ejemplo, IndexFlatL2 (distancia Euclidiana) es más simple.
    # Si los embeddings estuvieran normalizados a longitud unitaria, L2 y producto interno son equivalentes.
    index = faiss.IndexFlatL2(dim)
    index.add(vectors_np)
    faiss.write_index(index, str(index_path))

    # Crear el id_map: mapea índice de FAISS (0, 1, 2) a ID de BD (101, 102, 103)
    id_map_content = {i: doc_id for i, doc_id in enumerate(doc_ids_in_db)}
    with open(id_map_path, "wb") as f:
        pickle.dump(id_map_content, f)

    return index_path, id_map_path, all_predefined_embeddings, dim


# --- Tests ---
def test_faiss_retrieves_closest_doc(faiss_test_artefacts, monkeypatch):
    index_path, id_map_path, predefined_embeddings, dim = faiss_test_artefacts

    # Monkeypatch settings para que el retriever use los artefactos de test
    monkeypatch.setattr(global_settings, "index_path", str(index_path))
    monkeypatch.setattr(global_settings, "id_map_path", str(id_map_path))

    embedder = DummyEmbedder(dim=dim, predefined_embeddings=predefined_embeddings)
    # Nota: DenseFaissRetriever toma una instancia de SentenceTransformerEmbedder en su __init__.
    # Para este test unitario, es mejor si DenseFaissRetriever aceptara un EmbedderPort.
    # Si no, tendremos que mockear SentenceTransformerEmbedder o crear una subclase para el test.
    # ASUMAMOS por ahora que DenseFaissRetriever puede tomar cualquier EmbedderPort.
    # Si no, este es un punto a refactorizar en DenseFaissRetriever.
    
    # Refactor Propuesto para DenseFaissRetriever.__init__:
    # def __init__(self, embedder: EmbedderPort): # En lugar de SentenceTransformerEmbedder
    
    retriever = DenseFaissRetriever(embedder=embedder) # Pasamos nuestro DummyEmbedder

    query_text = "query_target_doc1" # Esta query está diseñada para estar cerca de doc1_text (ID 101)
    
    # Recordar que RetrieverPort ahora devuelve (ids, scores)
    retrieved_ids, retrieved_scores = retriever.retrieve(query_text, k=1)

    assert len(retrieved_ids) == 1
    assert retrieved_ids[0] == 101 # El ID de "doc1_text"
    assert len(retrieved_scores) == 1
    assert isinstance(retrieved_scores[0], float)
    # Para L2, el score (distancia) debería ser pequeño
    assert retrieved_scores[0] < 0.1 # Ajustar este umbral basado en los embeddings de prueba

def test_faiss_respects_k(faiss_test_artefacts, monkeypatch):
    index_path, id_map_path, predefined_embeddings, dim = faiss_test_artefacts
    monkeypatch.setattr(global_settings, "index_path", str(index_path))
    monkeypatch.setattr(global_settings, "id_map_path", str(id_map_path))

    embedder = DummyEmbedder(dim=dim, predefined_embeddings=predefined_embeddings)
    retriever = DenseFaissRetriever(embedder=embedder)

    query_text = "query_target_doc1" # Query que podría coincidir con varios si k es mayor
    k_val = 2
    retrieved_ids, retrieved_scores = retriever.retrieve(query_text, k=k_val)

    assert len(retrieved_ids) == k_val
    assert len(retrieved_scores) == k_val
    assert 101 in retrieved_ids # doc1 debería estar
    # El segundo podría ser doc2 o doc3 dependiendo de las distancias exactas

def test_faiss_no_match(faiss_test_artefacts, monkeypatch):
    index_path, id_map_path, predefined_embeddings, dim = faiss_test_artefacts
    monkeypatch.setattr(global_settings, "index_path", str(index_path))
    monkeypatch.setattr(global_settings, "id_map_path", str(id_map_path))

    embedder = DummyEmbedder(dim=dim, predefined_embeddings=predefined_embeddings)
    retriever = DenseFaissRetriever(embedder=embedder)

    query_text = "query_no_match" # Diseñada para no coincidir bien
    retrieved_ids, retrieved_scores = retriever.retrieve(query_text, k=1)

    assert isinstance(retrieved_ids, list)
    assert isinstance(retrieved_scores, list)
    assert len(retrieved_ids) == len(retrieved_scores)
    if retrieved_ids: # Si devuelve algo
        assert len(retrieved_ids) <=1
        # La distancia para "query_no_match" (vector de ceros) a los otros vectores será mayor
        # que la distancia de "query_target_doc1" a "doc1_text".
        # Puedes añadir un assert sobre el valor del score si es predecible.
        # Por ejemplo, si retrieved_scores[0] > algún umbral alto.


# Más tests posibles:
# - Qué pasa si k es mayor que el número de documentos en el índice.
# - Qué pasa con una query vacía (si el DummyEmbedder la maneja de forma específica).
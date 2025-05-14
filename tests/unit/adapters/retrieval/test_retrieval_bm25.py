# tests/unit/test_retrieval_bm25.py
import pytest
from src.adapters.retrieval.sparse_bm25 import SparseBM25Retriever # o el path correcto

# --- Fixtures (si son necesarias) ---
@pytest.fixture
def sample_documents_data() -> tuple[list[str], list[int]]:
    documents = [
        "El perro marrón rápido salta sobre el zorro perezoso.",
        "Nunca la lluvia en España cae principalmente en la llanura.",
        "El perro es el mejor amigo del hombre.",
        "Los zorros son ágiles y los perros también.",
    ]
    doc_ids = [10, 20, 30, 40] # IDs arbitrarios para los documentos
    return documents, doc_ids

# --- Tests ---
def test_bm25_retrieves_relevant_doc(sample_documents_data):
    documents, doc_ids = sample_documents_data
    retriever = SparseBM25Retriever(documents=documents, doc_ids=doc_ids)
    
    query = "perro amigo"
    # Asumimos que `retrieve` ahora devuelve (ids, scores)
    retrieved_ids, retrieved_scores = retriever.retrieve(query, k=1)
    
    assert len(retrieved_ids) == 1
    assert retrieved_ids[0] == 30 # "El perro es el mejor amigo del hombre."
    assert len(retrieved_scores) == 1
    assert isinstance(retrieved_scores[0], float) # O el tipo que esperes

def test_bm25_respects_k(sample_documents_data):
    documents, doc_ids = sample_documents_data
    retriever = SparseBM25Retriever(documents=documents, doc_ids=doc_ids)
    
    query = "LES"
    k_val = 2
    retrieved_ids, retrieved_scores = retriever.retrieve(query, k=k_val)
    
    assert len(retrieved_ids) == k_val
    assert len(retrieved_scores) == k_val
    # Podrías hacer aserciones más específicas sobre los IDs esperados si el orden es predecible

def test_bm25_no_match(sample_documents_data):
    documents, doc_ids = sample_documents_data
    retriever = SparseBM25Retriever(documents=documents, doc_ids=doc_ids)
    
    query = "gato unicornio inexistente"
    retrieved_ids, retrieved_scores = retriever.retrieve(query, k=1)
    
    # BM25 aún podría devolver algo si hay solapamiento de tokens comunes
    # o podría devolver una lista vacía si los scores son muy bajos o cero.
    # Esto depende de la implementación de rank_bm25 y cómo manejas scores cero.
    # Si se esperan resultados (incluso con scores bajos):
    # assert len(retrieved_ids) > 0 # o un número específico
    # Si se espera una lista vacía para no coincidencias fuertes:
    # assert len(retrieved_ids) == 0
    # assert len(retrieved_scores) == 0
    # Por ahora, seamos flexibles y solo verifiquemos la estructura:
    assert isinstance(retrieved_ids, list)
    assert isinstance(retrieved_scores, list)
    assert len(retrieved_ids) == len(retrieved_scores)


def test_bm25_empty_query(sample_documents_data):
    documents, doc_ids = sample_documents_data
    retriever = SparseBM25Retriever(documents=documents, doc_ids=doc_ids)
    
    query = ""
    retrieved_ids, retrieved_scores = retriever.retrieve(query, k=1)

    # rank_bm25 con query vacía usualmente da scores de 0 para todos los docs.
    # Devolvería todos los documentos con score 0 si k es lo suficientemente grande,
    # o los primeros k documentos con score 0.
    assert len(retrieved_ids) <= 1 # Debería devolver k documentos o menos
    assert len(retrieved_scores) <= 1
    if retrieved_scores: # Si devuelve algo
        assert all(score == 0.0 for score in retrieved_scores)

# Podrías añadir más tests:
# - Con documentos vacíos
# - Verificando el orden de los scores (el primer score debe ser >= al segundo, etc.)
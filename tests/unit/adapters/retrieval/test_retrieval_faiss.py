# tests/unit/test_retrieval_dense_faiss.py

import pytest
import numpy as np
import faiss
import pickle
from pathlib import Path

from src.settings import settings as global_settings
from src.adapters.retrieval.dense_faiss import DenseFaissRetriever
from src.core.ports import EmbedderPort


# --- Dummy Embedder for controlled testing ---
class DummyEmbedder(EmbedderPort):
    """A dummy embedder returning pre-defined vectors for given texts."""

    DIM: int

    def __init__(self, dim: int, predefined_embeddings: dict[str, list[float]] = None):
        self.DIM = dim
        self.predefined_embeddings = (
            predefined_embeddings if predefined_embeddings else {}
        )
        self.default_vector = [0.0] * self.DIM

    def embed(self, text: str) -> list[float]:
        return self.predefined_embeddings.get(text, self.default_vector)


# --- FAISS index and id-map artifact fixture ---
@pytest.fixture
def faiss_test_artifacts(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, list[float]], int]:
    """Creates a minimal FAISS index and ID map for dense retrieval tests."""
    dim = 4
    doc_embeddings = {
        "doc1_text": [1.0, 0.1, 0.2, 0.3],  # Closest to query_target_doc1
        "doc2_text": [0.2, 1.0, 0.3, 0.4],
        "doc3_text": [0.3, 0.2, 1.0, 0.5],
    }
    doc_ids_in_db = [101, 102, 103]

    query_embeddings = {
        "query_target_doc1": [0.9, 0.15, 0.25, 0.35],  # Closest to doc1_text
        "query_target_doc2": [0.25, 0.9, 0.35, 0.45],  # Closest to doc2_text
        "query_no_match": [0.0, 0.0, 0.0, 0.0],  # Not close to any doc
    }

    all_predefined_embeddings = {**doc_embeddings, **query_embeddings}

    index_path = tmp_path / "test_index.faiss"
    id_map_path = tmp_path / "test_id_map.pkl"

    vectors_np = np.array(list(doc_embeddings.values()), dtype="float32")
    index = faiss.IndexFlatL2(dim)
    index.add(vectors_np)
    faiss.write_index(index, str(index_path))

    id_map_content = {i: doc_id for i, doc_id in enumerate(doc_ids_in_db)}
    with open(id_map_path, "wb") as f:
        pickle.dump(id_map_content, f)

    return index_path, id_map_path, all_predefined_embeddings, dim


# --- Tests ---


def test_faiss_retrieves_closest_doc(faiss_test_artifacts, monkeypatch):
    """
    DenseFaissRetriever returns the closest document for a given query embedding.
    """
    index_path, id_map_path, predefined_embeddings, dim = faiss_test_artifacts
    monkeypatch.setattr(global_settings, "index_path", str(index_path))
    monkeypatch.setattr(global_settings, "id_map_path", str(id_map_path))

    embedder = DummyEmbedder(dim=dim, predefined_embeddings=predefined_embeddings)
    retriever = DenseFaissRetriever(embedder=embedder)

    query_text = "query_target_doc1"
    retrieved_ids, retrieved_scores = retriever.retrieve(query_text, k=1)

    assert len(retrieved_ids) == 1
    assert retrieved_ids[0] == 101  # "doc1_text"
    assert len(retrieved_scores) == 1
    assert isinstance(retrieved_scores[0], float)
    assert retrieved_scores[0] < 0.1  # Should be a small L2 distance


def test_faiss_respects_k(faiss_test_artifacts, monkeypatch):
    """
    DenseFaissRetriever returns exactly k results, including the most relevant.
    """
    index_path, id_map_path, predefined_embeddings, dim = faiss_test_artifacts
    monkeypatch.setattr(global_settings, "index_path", str(index_path))
    monkeypatch.setattr(global_settings, "id_map_path", str(id_map_path))

    embedder = DummyEmbedder(dim=dim, predefined_embeddings=predefined_embeddings)
    retriever = DenseFaissRetriever(embedder=embedder)

    query_text = "query_target_doc1"
    k_val = 2
    retrieved_ids, retrieved_scores = retriever.retrieve(query_text, k=k_val)

    assert len(retrieved_ids) == k_val
    assert len(retrieved_scores) == k_val
    assert 101 in retrieved_ids  # Closest doc should always be present


def test_faiss_no_match_returns_something(faiss_test_artifacts, monkeypatch):
    """
    Even for queries not close to any doc, returns k results (lowest relevance).
    """
    index_path, id_map_path, predefined_embeddings, dim = faiss_test_artifacts
    monkeypatch.setattr(global_settings, "index_path", str(index_path))
    monkeypatch.setattr(global_settings, "id_map_path", str(id_map_path))

    embedder = DummyEmbedder(dim=dim, predefined_embeddings=predefined_embeddings)
    retriever = DenseFaissRetriever(embedder=embedder)

    query_text = "query_no_match"
    retrieved_ids, retrieved_scores = retriever.retrieve(query_text, k=1)

    assert isinstance(retrieved_ids, list)
    assert isinstance(retrieved_scores, list)
    assert len(retrieved_ids) == len(retrieved_scores)
    if retrieved_ids:
        assert len(retrieved_ids) == 1
        assert (
            retrieved_scores[0] > 0.5
        )  # Arbitrary: Should be greater than "close match" threshold


def test_faiss_k_larger_than_docs(faiss_test_artifacts, monkeypatch):
    """
    If k is greater than number of docs in the index, returns all available docs.
    """
    index_path, id_map_path, predefined_embeddings, dim = faiss_test_artifacts
    monkeypatch.setattr(global_settings, "index_path", str(index_path))
    monkeypatch.setattr(global_settings, "id_map_path", str(id_map_path))

    embedder = DummyEmbedder(dim=dim, predefined_embeddings=predefined_embeddings)
    retriever = DenseFaissRetriever(embedder=embedder)

    query_text = "query_target_doc2"
    k_val = 5  # More than number of docs (3)
    retrieved_ids, retrieved_scores = retriever.retrieve(query_text, k=k_val)

    assert len(retrieved_ids) == 3  # Should return all docs present
    assert len(retrieved_scores) == 3


def test_faiss_empty_query_returns_default(faiss_test_artifacts, monkeypatch):
    """
    If the query is empty, uses the embedder's default vector and returns k results.
    """
    index_path, id_map_path, predefined_embeddings, dim = faiss_test_artifacts
    monkeypatch.setattr(global_settings, "index_path", str(index_path))
    monkeypatch.setattr(global_settings, "id_map_path", str(id_map_path))

    embedder = DummyEmbedder(dim=dim, predefined_embeddings=predefined_embeddings)
    retriever = DenseFaissRetriever(embedder=embedder)

    query_text = ""  # No embedding defined, uses default (zero vector)
    k_val = 2
    retrieved_ids, retrieved_scores = retriever.retrieve(query_text, k=k_val)

    assert len(retrieved_ids) == k_val
    assert len(retrieved_scores) == k_val
    # All returned scores should be the L2 distance to [0, 0, 0, 0] (can assert actual values if desired)

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from local_rag_backend.core.services.etl import ETLService
from local_rag_backend.core.services.rag_runtime import RagService
from local_rag_backend.infrastructure.embeddings import openai as openai_embedder_mod
from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder
from local_rag_backend.infrastructure.persistence.sql import (
    HistorySqlStorage,
    SqlDocumentStorage,
)
from local_rag_backend.infrastructure.persistence.sql.base import Base
from local_rag_backend.infrastructure.persistence.vector.storage import VectorStorage
from local_rag_backend.infrastructure.retrieval.dense_vector import DenseVectorRetriever
from local_rag_backend.infrastructure.retrieval.hybrid import HybridRetriever
from local_rag_backend.infrastructure.retrieval.sparse_bm25 import SparseBM25Retriever
from local_rag_backend.settings import settings


def _get_corpus_and_ids(doc_repo):
    docs = doc_repo.get_all_documents()
    return [d.content for d in docs], [d.id for d in docs]


class _DummyEmbeddingItem:
    def __init__(self, embedding):
        self.embedding = embedding


class _DummyEmbeddingsResp:
    def __init__(self, vectors):
        self.data = [_DummyEmbeddingItem(v) for v in vectors]


class _DummyEmbeddingsAPI:
    def create(self, model, input):
        # Tiny deterministic 4D embedding; good enough for FAISS L2.
        vecs = []
        for t in input:
            t = str(t)
            vecs.append(
                [
                    float(len(t)),
                    float(sum(ord(c) for c in t) % 17),
                    float(t.count("a")),
                    float(t.count("z")),
                ]
            )
        return _DummyEmbeddingsResp(vecs)


class _DummyOpenAI:
    def __init__(self, api_key):
        self.embeddings = _DummyEmbeddingsAPI()


class _DummyGen:
    def __init__(self, *a, **k):
        pass

    def generate(self, question, contexts):
        return f"answer:{question}:{len(contexts)}"


@pytest.mark.integration
def test_dense_and_hybrid_end_to_end(tmp_path, monkeypatch):
    # Settings
    db_path = tmp_path / "app.db"
    index_path = tmp_path / "idx.faiss"
    id_map_path = tmp_path / "id.json"

    monkeypatch.setattr(settings, "sqlite_url", f"sqlite:///{db_path}", raising=False)
    monkeypatch.setattr(settings, "index_path", str(index_path), raising=False)
    monkeypatch.setattr(settings, "id_map_path", str(id_map_path), raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "k", raising=False)
    monkeypatch.setattr(settings, "openai_embedding_model", "dummy-4", raising=False)
    monkeypatch.setattr(openai_embedder_mod, "_MODEL_DIM", {"dummy-4": 4}, raising=False)
    monkeypatch.setattr(openai_embedder_mod, "OpenAI", _DummyOpenAI, raising=True)

    # DB setup
    engine = create_engine(settings.sqlite_url, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    doc_repo = SqlDocumentStorage(session_factory=SessionLocal)
    history_repo = HistorySqlStorage(session_factory=SessionLocal)

    embedder = OpenAIEmbedder(model="dummy-4")
    vec_repo = VectorStorage(
        index_path=str(index_path), id_map_path=str(id_map_path), dim=embedder.dim
    )

    etl = ETLService(doc_repo, vec_repo, embedder)
    ids = etl.ingest(["alpha alpha alpha", "zzzz zzzz zzzz"])
    assert len(list(ids)) == 2

    dense = DenseVectorRetriever(embedder=embedder, vector_repo=vec_repo, doc_repo=doc_repo)
    docs, scores = dense.retrieve("alpha", k=1)
    assert len(docs) == 1
    assert docs[0].content.startswith("alpha")
    assert len(scores) == 1

    # Hybrid: ensure sparse is wired and returns something too.
    corpus, doc_ids = _get_corpus_and_ids(doc_repo)
    sparse = SparseBM25Retriever(documents=corpus, doc_ids=doc_ids, doc_repo=doc_repo)
    hybrid = HybridRetriever(dense=dense, sparse=sparse, alpha=0.5)

    svc = RagService(retriever=hybrid, generator=_DummyGen(), history_storage=history_repo)
    resp = svc.ask("alpha", top_k=1)
    assert resp["answer"].startswith("answer:alpha:1")
    assert resp["docs"]

from __future__ import annotations

import rank_bm25

from local_rag_backend.composition.adapters import build_retriever_from_settings
from local_rag_backend.composition.container import AppContainer
from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.domain.retrieval import RetrievalFilter, RetrievalRequest
from local_rag_backend.core.domain.types import DocId
from local_rag_backend.core.use_cases.docs_mutation import (
    MutationCoordinator,
    MutationIntent,
    MutationUpsertInput,
)
from local_rag_backend.settings import Settings


def test_sparse_bm25_is_built_once_for_unfiltered_service_queries(monkeypatch) -> None:
    builds = 0
    real_bm25 = rank_bm25.BM25Okapi

    class _CountingBM25(real_bm25):
        def __init__(self, corpus):
            nonlocal builds
            builds += 1
            super().__init__(corpus)

    monkeypatch.setattr(rank_bm25, "BM25Okapi", _CountingBM25)
    docs = [
        Document(DocId("one"), "capital france paris", metadata={"scope": "public"}),
        Document(DocId("two"), "python auth guard", metadata={"scope": "public"}),
    ]

    class _Repo:
        def get_all_documents(self):
            return docs

        def get(self, ids):
            return [doc for doc in docs if doc.id in ids]

    retriever = build_retriever_from_settings(
        settings_obj=Settings(),
        retrieval_mode="sparse",
        doc_repo=_Repo(),
        dense_embedder_factory=lambda: None,
    )
    retriever.retrieve(RetrievalRequest(query="france", top_k=1))
    retriever.retrieve(RetrievalRequest(query="python", top_k=1))

    assert builds == 1

    retriever.retrieve(
        RetrievalRequest(
            query="france",
            top_k=1,
            filters=(RetrievalFilter("scope", ("public",)),),
        )
    )
    assert builds == 2  # Filtered queries still build a filtered BM25 index.


def test_warmed_postings_are_rebuilt_after_document_mutation(tmp_path) -> None:
    class Generator:
        def generate(self, question, contexts):
            raise AssertionError("Retrieval acceptance must not invoke a model")

    container = AppContainer(
        settings_obj=Settings(
            sqlite_url=f"sqlite:///{tmp_path / 'corpus.sqlite3'}",
            data_dir=tmp_path / "data",
            retrieval_mode="sparse",
            ollama_enabled=True,
            openai_api_key=None,
            disable_embedding_cache=True,
        ),
        ollama_generator_factory=lambda **_kwargs: Generator(),
    )
    try:
        container.initialize()
        coordinator = MutationCoordinator(
            settings_obj=container.settings_obj, ports=container.docs_mutation_ports()
        )
        coordinator.execute(
            MutationIntent(
                op_id="bm25-acceptance-initial",
                upserts=(
                    MutationUpsertInput(external_id="one", content="alpha initial"),
                    MutationUpsertInput(external_id="two", content="beta initial"),
                ),
            )
        )
        old = container.get_rag_service()
        request = RetrievalRequest(query="alpha", top_k=1, mode="sparse")
        assert old.retriever.retrieve(request).documents[0].content == "alpha initial"
        coordinator.execute(
            MutationIntent(
                op_id="bm25-acceptance-update",
                upserts=(MutationUpsertInput(external_id="one", content="omega updated"),),
            )
        )
        current = container.get_rag_service()
        assert current is not old
        assert (
            current.retriever.retrieve(RetrievalRequest(query="omega", top_k=1, mode="sparse"))
            .documents[0]
            .content
            == "omega updated"
        )
        assert old.retriever.retrieve(request).documents[0].content == "alpha initial"
    finally:
        container.close()

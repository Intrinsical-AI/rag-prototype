from __future__ import annotations

from types import SimpleNamespace

from local_rag_backend.composition.adapters import _DefaultRagRuntimeFactory
from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.domain.types import DocId


def test_ask_eval_dense_does_not_preload_the_corpus() -> None:
    docs = [Document(DocId("one"), "content")]
    calls = 0
    preloaded: list[object] = []

    def _all_documents():
        nonlocal calls
        calls += 1
        return docs

    def _build_retriever(_cfg, _repo, *, preloaded_docs):
        preloaded.append(preloaded_docs)
        return object()

    class _Service:
        def ask(self, **_kwargs):
            return {"answer": "ok", "docs": [], "scores": []}

    factory = _DefaultRagRuntimeFactory(
        doc_repo_factory=lambda: SimpleNamespace(get_all_documents=_all_documents),
        history_repo_factory=lambda: object(),
        build_retriever_from_config=_build_retriever,
        build_generator_from_config=lambda _cfg: object(),
        rag_service_factory=lambda *_args: _Service(),
    )
    cfg = SimpleNamespace(retrieval_mode="dense", k=1, filters=None, dual_candidate_k=None)

    assert factory.run_ask_eval(question="query", cfg=cfg)["answer"] == "ok"
    assert calls == 0
    assert preloaded == [None]

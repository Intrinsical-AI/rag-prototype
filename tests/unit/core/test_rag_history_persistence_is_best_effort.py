from local_rag_backend.core.services.rag_runtime import NO_DOCS_ANSWER, RagService


class _DummyRetriever:
    def __init__(self, docs, scores):
        self._docs = docs
        self._scores = scores

    def retrieve(self, query, k=5):
        return self._docs[:k], self._scores[:k]


class _DummyGenerator:
    def generate(self, question, contexts):
        return f"ans:{question}:{len(contexts)}"


class _FailingHistory:
    def save(self, q, a, source_ids):
        raise RuntimeError("history-down")


def test_rag_history_failure_does_not_break_answer():
    from local_rag_backend.core.domain.entities import Document

    svc = RagService(
        retriever=_DummyRetriever([Document(id=1, content="c")], [1.0]),
        generator=_DummyGenerator(),
        history_storage=_FailingHistory(),
    )

    out = svc.ask("q", top_k=1)
    assert out["answer"].startswith("ans:q:1")
    assert out["docs"]


def test_rag_history_failure_does_not_break_empty_retrieval():
    svc = RagService(
        retriever=_DummyRetriever([], []),
        generator=_DummyGenerator(),
        history_storage=_FailingHistory(),
    )
    out = svc.ask("q", top_k=1)
    assert out["answer"] == NO_DOCS_ANSWER

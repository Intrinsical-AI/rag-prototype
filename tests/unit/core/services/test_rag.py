from src.core.domain.entities import Document
from src.core.services.rag import RagService


class DummyRetriever:
    def __init__(self, docs, scores):
        self._docs, self._scores = docs, scores
        self.last_query = None

    def retrieve(self, query, k=3):
        self.last_query = query
        return self._docs[:k], self._scores[:k]


class DummyGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, question, contexts):
        self.calls.append((question, contexts))
        return "dummy-answer-for:" + question


class DummyHistory:
    def __init__(self):
        self.saved = []

    def save(self, q, a, source_ids):
        self.saved.append((q, a, list(source_ids)))


def test_rag_service_flow():
    doc = Document(id=1, content="contenido relevante")
    retriever = DummyRetriever([doc], [0.85])
    generator = DummyGenerator()
    history = DummyHistory()

    rag = RagService(retriever, generator, history)

    resp = rag.ask("¿Qué es esto?", top_k=1)
    # Comprobamos respuesta
    assert resp["answer"].startswith("dummy-answer-for")
    assert resp["docs"] == [doc]
    assert resp["scores"] == [0.85]
    # ¿Se guardó en history?
    assert history.saved == [("¿Qué es esto?", resp["answer"], [1])]


def test_rag_service_empty_docs():
    retriever = DummyRetriever([], [])
    generator = DummyGenerator()
    history = DummyHistory()
    rag = RagService(retriever, generator, history)
    resp = rag.ask("vacío", top_k=2)
    assert resp["answer"].startswith("No hay documentos indexados")
    assert resp["docs"] == []
    assert resp["scores"] == []

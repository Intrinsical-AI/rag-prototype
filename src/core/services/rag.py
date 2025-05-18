# src/core/rag.py

from typing import Any, Mapping

from src.core.ports import GeneratorPort, QAHistoryPort, RetrieverPort


class RagService:
    def __init__(
        self, retriever: RetrieverPort, generator: GeneratorPort, history: QAHistoryPort
    ):
        self.retriever = retriever
        self.generator = generator
        self.history = history

    def ask(self, question: str, top_k: int = 3) -> Mapping[str, Any]:
        docs, scores = self.retriever.retrieve(question, top_k)
        if not docs:
            return {
                "answer": "No hay documentos indexados. Por favor, ejecuta la ingestión.",
                "docs": [],
                "scores": [],
            }
        answer = self.generator.generate(question, [d.content for d in docs])
        self.history.save(question, answer, [d.id for d in docs])
        return {"answer": answer, "docs": docs, "scores": scores}

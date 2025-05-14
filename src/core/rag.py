# === file: src/core/rag.py ===
"""Servicio principal de Retrieval‑Augmented Generation"""
from __future__ import annotations

from typing import List, Dict, Any

from sqlalchemy.orm import Session

from src.core.ports import RetrieverPort, GeneratorPort
from src.db import crud

__all__ = ["RagService"]


class RagService:
    """Orquesta recuperación + generación."""

    def __init__(self, retriever: RetrieverPort, generator: GeneratorPort):
        self.retriever = retriever
        self.generator = generator

    # k default 3 como requieren los tests
    def ask(self, db: Session, question: str, *, k: int = 3) -> Dict[str, Any]:
        doc_ids, _ = self.retriever.retrieve(question, k)
        documents = crud.get_documents(db, doc_ids) if doc_ids else []
        contexts = [d.content for d in documents]
        answer = self.generator.generate(question, contexts)

        crud.save_qa_history(db, question, answer)
        return {"answer": answer, "source_ids": doc_ids}

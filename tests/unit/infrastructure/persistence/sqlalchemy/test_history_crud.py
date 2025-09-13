# tests/unit/infrastructure/persistence/sqlalchemy/test_history_crud.py
import pytest

from local_rag_backend.infrastructure.persistence.sqlalchemy import models  # noqa: F401
from local_rag_backend.infrastructure.persistence.sqlalchemy.crud import (
    get_history,
    save_qa_history,
)


@pytest.mark.usefixtures("in_memory_sqlite")
def test_add_and_get_history_basic(in_memory_sqlite):
    session_factory = in_memory_sqlite
    # Ensure tables exist on this engine (should already be created by fixture)
    session = session_factory()
    try:
        # Insert two records
        save_qa_history(session, "Q1?", "A1", source_ids=[1, 2])
        save_qa_history(session, "Q2?", "A2", source_ids=None)

        # Retrieve history
        entries = get_history(session, limit=10, offset=0)
        assert len(entries) == 2
        texts = {(e.question, e.answer) for e in entries}
        assert {("Q1?", "A1"), ("Q2?", "A2")} == texts

        # Validate source_ids typing
        # Note: JSON(None) may come back as None
        id_map = {e.question: (e.source_ids or []) for e in entries}
        assert id_map["Q1?"] == [1, 2]
        assert id_map["Q2?"] == []
    finally:
        session.close()

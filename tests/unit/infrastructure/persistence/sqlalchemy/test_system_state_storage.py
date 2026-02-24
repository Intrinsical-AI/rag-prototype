from __future__ import annotations

import pytest
from sqlalchemy import text

from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SystemStateStorage


def test_system_state_get_version_defaults_to_zero(in_memory_sqlite) -> None:
    st = SystemStateStorage(session_factory=in_memory_sqlite)
    assert st.get_version("rag_service") == 0


def test_system_state_bump_version_is_monotonic(in_memory_sqlite) -> None:
    st = SystemStateStorage(session_factory=in_memory_sqlite)

    assert st.bump_version("rag_service") == 1
    assert st.bump_version("rag_service") == 2
    assert st.get_version("rag_service") == 2


def test_system_state_storage_creates_table_on_demand(in_memory_sqlite) -> None:
    st = SystemStateStorage(session_factory=in_memory_sqlite)
    st.bump_version("rag_service")

    with in_memory_sqlite() as session:
        row = session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='system_state'")
        ).first()
        assert row is not None


def test_system_state_storage_rejects_blank_keys(in_memory_sqlite) -> None:
    st = SystemStateStorage(session_factory=in_memory_sqlite)

    with pytest.raises(ValueError, match="must not be blank"):
        st.get_version("   ")

    with pytest.raises(ValueError, match="must not be blank"):
        st.bump_version("")

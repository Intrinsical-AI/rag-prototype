"""Tests for core domain identity helpers (types.py)."""

from __future__ import annotations

import uuid

from local_rag_backend.core.domain.types import new_doc_id


def test_new_doc_id_default_prefix() -> None:
    doc_id = new_doc_id()
    assert doc_id.startswith("doc:")


def test_new_doc_id_custom_prefix() -> None:
    doc_id = new_doc_id(prefix="chunk")
    assert doc_id.startswith("chunk:")


def test_new_doc_id_returns_str() -> None:
    # DocId is a NewType of str — the runtime value is a plain str
    assert isinstance(new_doc_id(), str)


def test_new_doc_id_suffix_is_valid_uuid() -> None:
    doc_id = new_doc_id(prefix="x")
    _, suffix = doc_id.split(":", 1)
    parsed = uuid.UUID(suffix)
    assert parsed.version == 7


def test_new_doc_id_is_unique() -> None:
    ids = {new_doc_id() for _ in range(50)}
    assert len(ids) == 50

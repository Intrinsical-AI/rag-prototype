"""Tests for looks_like_pickle and id_map_json helpers."""

from __future__ import annotations

import pickle

from local_rag_backend.infrastructure.persistence.shared.id_map_json import looks_like_pickle


def test_looks_like_pickle_rejects_pickle_v2_bytes() -> None:
    data = pickle.dumps({"key": "value"}, protocol=2)
    assert looks_like_pickle(data) is True


def test_looks_like_pickle_rejects_pickle_v4_bytes() -> None:
    data = pickle.dumps([1, 2, 3], protocol=4)
    assert looks_like_pickle(data) is True


def test_looks_like_pickle_accepts_json_bytes() -> None:
    assert looks_like_pickle(b'["doc:1", "doc:2"]') is False


def test_looks_like_pickle_accepts_empty_bytes() -> None:
    assert looks_like_pickle(b"") is False


def test_looks_like_pickle_accepts_plain_text() -> None:
    assert looks_like_pickle(b"hello world") is False


def test_looks_like_pickle_boundary_single_byte() -> None:
    # 0x80 alone should still trigger the guard
    assert looks_like_pickle(b"\x80") is True
    assert looks_like_pickle(b"\x7f") is False

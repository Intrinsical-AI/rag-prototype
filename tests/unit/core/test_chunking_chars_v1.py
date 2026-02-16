# tests/unit/core/test_chunking_chars_v1.py

from __future__ import annotations

import pytest

from local_rag_backend.core.services.chunking import chunk_chars_v1


def test_chunk_chars_v1_is_deterministic():
    text = "0123456789ABCDEFGHIJ"  # 20
    a = chunk_chars_v1(text, max_chars=7, overlap=2)
    b = chunk_chars_v1(text, max_chars=7, overlap=2)
    assert a == b


@pytest.mark.parametrize(
    "text,max_chars,overlap",
    [
        ("", 10, 3),
        ("short", 10, 3),
        ("0123456789" * 3 + "END", 10, 10),  # overlap == max_chars (clamped)
        ("abcdefghijklmnopqrstuvwxyz", 5, 10),  # overlap > max_chars (clamped)
        ("A" * 25, 3, 0),
    ],
)
def test_chunk_chars_v1_boundaries_and_progress(text: str, max_chars: int, overlap: int):
    chunks = chunk_chars_v1(text, max_chars=max_chars, overlap=overlap)
    assert chunks
    assert chunks[0].chunk_index == 0

    # Boundaries are consistent with text slicing.
    for c in chunks:
        assert c.text == text[c.start_char : c.end_char]
        assert 0 <= c.start_char <= c.end_char <= len(text)

    # Forward progress (except trivial single-chunk cases).
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.start_char > prev.start_char

    # Reconstruct the last character when text is non-empty.
    if text:
        assert chunks[-1].end_char == len(text)
        assert chunks[-1].text[-1] == text[-1]


def test_chunk_chars_v1_overlap_exact():
    text = "0123456789ABCDEFGHIJ"  # 20
    max_chars = 8
    overlap = 3
    chunks = chunk_chars_v1(text, max_chars=max_chars, overlap=overlap)
    assert len(chunks) >= 2

    step = max_chars - overlap
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        if i == 0:
            assert c.start_char == 0
        else:
            assert c.start_char == chunks[i - 1].start_char + step

    # Adjacent overlap length is exactly `overlap` while not at end.
    for a, b in zip(chunks, chunks[1:], strict=False):
        overlapped = text[b.start_char : a.end_char]
        if a.end_char < len(text):
            assert len(overlapped) == overlap

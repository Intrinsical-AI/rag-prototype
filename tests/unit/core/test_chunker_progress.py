# tests/unit/core/test_chunker_progress.py
import pytest

from local_rag_backend.core.services.ingestion import default_chunker


@pytest.mark.parametrize(
    "text,max_chars,overlap",
    [
        ("0123456789" * 3 + "END", 10, 10),            # overlap == max_chars
        ("abcdefghijklmnopqrstuvwxyz", 5, 10),           # overlap > max_chars (clamped)
        ("short", 10, 5),                                # text shorter than window
        ("😀🚀✨ unicode test 😀", 4, 2),                 # unicode / multi-byte
        ("", 8, 4),                                      # empty text
        ("A" * 25, 3, 0),                                # zero overlap small window
    ],
)
def test_default_chunker_forward_progress(text, max_chars, overlap):
    chunk = default_chunker(max_chars=max_chars, overlap=overlap)
    chunks = chunk(text, None)

    assert isinstance(chunks, list)

    if not text:
        # For empty text, expect a single empty chunk or no chunks depending on logic
        # Our implementation returns [""] if len(text) <= max_chars
        assert len(chunks) == 1
        assert chunks[0] == ""
        return

    # Last chunk should end with the last character of the original text
    assert chunks[-1][-1] == text[-1]

    # If multiple chunks exist, ensure incremental forward progress and overlap
    if len(chunks) > 1:
        for a, b in zip(chunks, chunks[1:], strict=False):
            assert a, "chunk must be non-empty"
            assert b, "chunk must be non-empty"
            assert a[-1] in b, "Expected overlap between adjacent chunks"

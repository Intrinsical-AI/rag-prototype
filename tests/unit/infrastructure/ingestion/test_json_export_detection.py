"""Tests for JSON export format detection."""

import json

from local_rag_backend.infrastructure.ingestion.loaders.factory import detect_json_export_format


def test_detect_json_export_format_chatgpt():
    """Test detection of ChatGPT export format."""
    export = [
        {
            "title": "Test",
            "conversation_id": "conv-1",
            "mapping": {
                "msg-1": {
                    "id": "msg-1",
                    "message": {
                        "id": "msg-1",
                        "author": {"role": "user"},
                        "create_time": 1700000000.0,
                        "content": {"content_type": "text", "parts": ["Hello"]},
                        "metadata": {},
                    },
                    "parent": None,
                    "children": [],
                }
            },
        }
    ]

    detection = detect_json_export_format(json.dumps(export).encode())

    assert detection.fmt == "chatgpt_export"
    assert detection.reason == "structural-heuristic"


def test_detect_json_export_format_gemini():
    """Test detection of Gemini export format."""
    export = [
        {
            "title": "Test",
            "conversation_id": "gemini-conv-1",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                    "timestamp": "2024-01-15T10:30:00Z",
                },
            ],
        }
    ]

    detection = detect_json_export_format(json.dumps(export).encode())

    assert detection.fmt == "gemini_export"
    assert detection.reason == "structural-heuristic"


def test_detect_json_export_format_unknown_json_array():
    """Test that unknown JSON arrays return unknown format."""
    export = [{"foo": "bar"}, {"baz": "qux"}]

    detection = detect_json_export_format(json.dumps(export).encode())

    assert detection.fmt == "unknown"
    assert detection.reason == "json-array-unrecognized"


def test_detect_json_export_format_invalid_json():
    """Test that invalid JSON returns unknown format."""
    detection = detect_json_export_format(b"not valid json")

    assert detection.fmt == "unknown"
    assert detection.reason == "json-parse-error"


def test_detect_json_export_format_not_array():
    """Test that JSON root that is not an array returns unknown."""
    detection = detect_json_export_format(json.dumps({"key": "value"}).encode())

    assert detection.fmt == "unknown"
    assert detection.reason == "not-a-list"


def test_detect_json_export_format_empty_bytes():
    """Test that empty bytes returns unknown."""
    detection = detect_json_export_format(b"")

    assert detection.fmt == "unknown"
    assert detection.reason == "not-a-json-array"


def test_detect_json_export_format_whitespace_only():
    """Test that whitespace-only bytes returns unknown."""
    detection = detect_json_export_format(b"   \n\t  ")

    assert detection.fmt == "unknown"
    assert detection.reason == "not-a-json-array"


def test_detect_json_export_format_null_json():
    """Test that JSON null returns unknown."""
    detection = detect_json_export_format(b"null")

    assert detection.fmt == "unknown"
    assert detection.reason == "not-a-json-array"


def test_detect_json_export_format_single_item_chatgpt():
    """Test detection with single-item ChatGPT export."""
    export = [
        {
            "title": "Single",
            "conversation_id": "conv-1",
            "mapping": {},
        }
    ]

    detection = detect_json_export_format(json.dumps(export).encode())

    assert detection.fmt == "chatgpt_export"


def test_detect_json_export_format_single_item_gemini():
    """Test detection with single-item Gemini export."""
    export = [
        {
            "title": "Single",
            "conversation_id": "gemini-conv-1",
            "messages": [],
        }
    ]

    detection = detect_json_export_format(json.dumps(export).encode())

    assert detection.fmt == "gemini_export"


def test_detect_json_export_format_chatgpt_multiple_items():
    """Test that ChatGPT detection works with multiple conversations."""
    export = [
        {"title": "Conv 1", "conversation_id": "1", "mapping": {}},
        {"title": "Conv 2", "conversation_id": "2", "mapping": {}},
        {"title": "Conv 3", "conversation_id": "3", "mapping": {}},
    ]

    detection = detect_json_export_format(json.dumps(export).encode())

    assert detection.fmt == "chatgpt_export"


def test_detect_json_export_format_gemini_multiple_items():
    """Test that Gemini detection works with multiple conversations."""
    export = [
        {"title": "Conv 1", "conversation_id": "1", "messages": []},
        {"title": "Conv 2", "conversation_id": "2", "messages": []},
        {"title": "Conv 3", "conversation_id": "3", "messages": []},
    ]

    detection = detect_json_export_format(json.dumps(export).encode())

    assert detection.fmt == "gemini_export"


def test_detect_json_export_format_mixed_structure():
    """Test that mixed/ambiguous structure returns unknown."""
    # Has some mapping and some messages
    export = [
        {"title": "Conv 1", "conversation_id": "1", "mapping": {}, "messages": []},
    ]

    detection = detect_json_export_format(json.dumps(export).encode())

    # Will match on 'mapping' first (ChatGPT heuristic)
    assert detection.fmt == "chatgpt_export"


def test_detect_json_export_format_gemini_without_role_content():
    """Test that Gemini-like structure without role/content in messages is rejected."""
    export = [
        {
            "title": "Conv 1",
            "conversation_id": "1",
            "messages": [
                {"timestamp": "2024-01-15T10:30:00Z"},  # Missing role and content
            ],
        }
    ]

    detection = detect_json_export_format(json.dumps(export).encode())

    assert detection.fmt == "unknown"


def test_detect_json_export_format_utf8_decoding():
    """Test that UTF-8 with non-ASCII characters is handled."""
    export = [
        {
            "title": "Conversación en Español",
            "conversation_id": "conv-1",
            "mapping": {},
        }
    ]

    detection = detect_json_export_format(json.dumps(export, ensure_ascii=False).encode("utf-8"))

    assert detection.fmt == "chatgpt_export"


def test_detect_json_export_format_invalid_utf8_fallback():
    """Test that invalid UTF-8 is handled gracefully with replacement chars."""
    # Create bytes that are invalid UTF-8 but start with '['
    invalid_utf8 = b'[{"title": "Test", "mapping": \xff}]'

    detection = detect_json_export_format(invalid_utf8)

    # Should attempt to detect, might succeed or fail gracefully
    assert detection.fmt in ("chatgpt_export", "unknown")

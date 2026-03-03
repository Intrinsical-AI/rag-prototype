"""Tests for GeminiLoader."""

import json

import pytest

from local_rag_backend.infrastructure.ingestion.loaders.gemini_loader import GeminiLoader


def _make_gemini_export(conversations: list[dict]) -> bytes:
    """Helper to create a Gemini export JSON."""
    return json.dumps(conversations).encode("utf-8")


def test_gemini_loader_minimal_valid_export():
    """Test minimal Gemini export with 1 conversation, 2 messages."""
    export = [
        {
            "title": "Test Conversation",
            "conversation_id": "gemini-conv-123",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello Gemini",
                    "timestamp": "2024-01-15T10:30:00Z",
                },
                {
                    "role": "model",
                    "content": "Hello! I'm Gemini.",
                    "timestamp": "2024-01-15T10:30:05Z",
                },
            ],
        }
    ]

    loader = GeminiLoader(_make_gemini_export(export))
    items = list(loader.load())

    assert len(items) == 2
    assert items[0].text == "Hello Gemini"
    assert items[0].metadata["conversation_id"] == "gemini-conv-123"
    assert items[0].metadata["role"] == "user"
    assert items[0].metadata["title"] == "Test Conversation"
    assert items[0].metadata["timestamp"] == "2024-01-15T10:30:00Z"

    assert items[1].text == "Hello! I'm Gemini."
    assert items[1].metadata["role"] == "model"


def test_gemini_loader_filters_empty_content():
    """Test that messages with empty content are filtered."""
    export = [
        {
            "title": "Test",
            "conversation_id": "gemini-conv-1",
            "messages": [
                {
                    "role": "user",
                    "content": "",
                    "timestamp": "2024-01-15T10:30:00Z",
                },
                {
                    "role": "user",
                    "content": "   ",
                    "timestamp": "2024-01-15T10:30:05Z",
                },
                {
                    "role": "model",
                    "content": "Valid response",
                    "timestamp": "2024-01-15T10:30:10Z",
                },
            ],
        }
    ]

    loader = GeminiLoader(_make_gemini_export(export))
    items = list(loader.load())

    assert len(items) == 1
    assert items[0].text == "Valid response"


def test_gemini_loader_filters_null_content():
    """Test that messages with None content are filtered."""
    export = [
        {
            "title": "Test",
            "conversation_id": "gemini-conv-1",
            "messages": [
                {
                    "role": "user",
                    "content": None,
                    "timestamp": "2024-01-15T10:30:00Z",
                },
                {
                    "role": "model",
                    "content": "Valid",
                    "timestamp": "2024-01-15T10:30:05Z",
                },
            ],
        }
    ]

    loader = GeminiLoader(_make_gemini_export(export))
    items = list(loader.load())

    assert len(items) == 1
    assert items[0].text == "Valid"


def test_gemini_loader_multiple_conversations():
    """Test multiple conversations are all yielded."""
    export = [
        {
            "title": "Conv 1",
            "conversation_id": "gemini-conv-1",
            "messages": [
                {
                    "role": "user",
                    "content": "Message 1",
                    "timestamp": "2024-01-15T10:30:00Z",
                },
            ],
        },
        {
            "title": "Conv 2",
            "conversation_id": "gemini-conv-2",
            "messages": [
                {
                    "role": "user",
                    "content": "Message 2",
                    "timestamp": "2024-01-15T10:30:00Z",
                },
            ],
        },
    ]

    loader = GeminiLoader(_make_gemini_export(export))
    items = list(loader.load())

    assert len(items) == 2
    assert items[0].metadata["conversation_id"] == "gemini-conv-1"
    assert items[0].metadata["title"] == "Conv 1"
    assert items[1].metadata["conversation_id"] == "gemini-conv-2"
    assert items[1].metadata["title"] == "Conv 2"


def test_gemini_loader_invalid_json_raises_value_error():
    """Test that invalid JSON raises ValueError."""
    with pytest.raises(ValueError, match="invalid JSON"):
        loader = GeminiLoader(b"not json")
        list(loader.load())


def test_gemini_loader_root_not_list_raises_value_error():
    """Test that root not being a list raises ValueError."""
    with pytest.raises(ValueError, match="expected a JSON array"):
        loader = GeminiLoader(json.dumps({"key": "value"}).encode())
        list(loader.load())


def test_gemini_loader_accepts_str_input():
    """Test that constructor accepts str as well as bytes."""
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

    loader = GeminiLoader(json.dumps(export))  # Pass as str
    items = list(loader.load())

    assert len(items) == 1
    assert items[0].text == "Hello"


def test_gemini_loader_empty_array_yields_nothing():
    """Test that an empty export array yields no items."""
    loader = GeminiLoader(b"[]")
    items = list(loader.load())

    assert items == []


def test_gemini_loader_missing_messages_field():
    """Test that conversations without messages field are skipped."""
    export = [
        {
            "title": "Conv without messages",
            "conversation_id": "gemini-conv-1",
            # No 'messages' field
        },
        {
            "title": "Conv with messages",
            "conversation_id": "gemini-conv-2",
            "messages": [
                {
                    "role": "user",
                    "content": "Valid",
                    "timestamp": "2024-01-15T10:30:00Z",
                },
            ],
        },
    ]

    loader = GeminiLoader(_make_gemini_export(export))
    items = list(loader.load())

    assert len(items) == 1
    assert items[0].text == "Valid"


def test_gemini_loader_non_dict_messages():
    """Test that non-dict message items are skipped."""
    export = [
        {
            "title": "Test",
            "conversation_id": "gemini-conv-1",
            "messages": [
                "not a dict",
                {
                    "role": "user",
                    "content": "Valid",
                    "timestamp": "2024-01-15T10:30:00Z",
                },
            ],
        }
    ]

    loader = GeminiLoader(_make_gemini_export(export))
    items = list(loader.load())

    assert len(items) == 1
    assert items[0].text == "Valid"


def test_gemini_loader_metadata_includes_all_fields():
    """Test that all expected metadata fields are present."""
    export = [
        {
            "title": "My Title",
            "conversation_id": "gemini-conv-xyz",
            "messages": [
                {
                    "role": "user",
                    "content": "Content here",
                    "timestamp": "2024-01-15T10:30:00Z",
                },
            ],
        }
    ]

    loader = GeminiLoader(_make_gemini_export(export))
    items = list(loader.load())

    assert len(items) == 1
    metadata = items[0].metadata
    assert metadata["source"] == "gemini_export"
    assert metadata["conversation_id"] == "gemini-conv-xyz"
    assert metadata["title"] == "My Title"
    assert metadata["role"] == "user"
    assert metadata["timestamp"] == "2024-01-15T10:30:00Z"

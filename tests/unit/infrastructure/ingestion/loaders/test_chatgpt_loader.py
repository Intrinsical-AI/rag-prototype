"""Tests for ChatGPTLoader."""

import json

import pytest

from local_rag_backend.infrastructure.ingestion.loaders.chatgpt_loader import ChatGPTLoader


def _make_chatgpt_export(conversations: list[dict]) -> bytes:
    """Helper to create a ChatGPT export JSON."""
    return json.dumps(conversations).encode("utf-8")


def test_chatgpt_loader_minimal_valid_export():
    """Test minimal ChatGPT export with 1 conversation, 2 messages."""
    export = [
        {
            "title": "Test Conversation",
            "conversation_id": "conv-123",
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
                    "children": ["msg-2"],
                },
                "msg-2": {
                    "id": "msg-2",
                    "message": {
                        "id": "msg-2",
                        "author": {"role": "assistant"},
                        "create_time": 1700000010.0,
                        "content": {"content_type": "text", "parts": ["Hi there"]},
                        "metadata": {"model_slug": "gpt-4o"},
                    },
                    "parent": "msg-1",
                    "children": [],
                },
            },
        }
    ]

    loader = ChatGPTLoader(_make_chatgpt_export(export))
    items = list(loader.load())

    assert len(items) == 2
    assert items[0].text == "Hello"
    assert items[0].metadata["conversation_id"] == "conv-123"
    assert items[0].metadata["role"] == "user"
    assert items[0].metadata["message_id"] == "msg-1"
    assert items[0].metadata.get("model_slug") is None

    assert items[1].text == "Hi there"
    assert items[1].metadata["role"] == "assistant"
    assert items[1].metadata["model_slug"] == "gpt-4o"


def test_chatgpt_loader_filters_non_text_content_type():
    """Test that non-text content types are filtered."""
    export = [
        {
            "title": "Mixed content",
            "conversation_id": "conv-1",
            "mapping": {
                "msg-1": {
                    "id": "msg-1",
                    "message": {
                        "id": "msg-1",
                        "author": {"role": "user"},
                        "create_time": 1700000000.0,
                        "content": {"content_type": "code", "parts": ["def foo(): pass"]},
                        "metadata": {},
                    },
                    "parent": None,
                    "children": [],
                },
                "msg-2": {
                    "id": "msg-2",
                    "message": {
                        "id": "msg-2",
                        "author": {"role": "assistant"},
                        "create_time": 1700000010.0,
                        "content": {"content_type": "text", "parts": ["Valid text"]},
                        "metadata": {},
                    },
                    "parent": None,
                    "children": [],
                },
            },
        }
    ]

    loader = ChatGPTLoader(_make_chatgpt_export(export))
    items = list(loader.load())

    assert len(items) == 1
    assert items[0].text == "Valid text"


def test_chatgpt_loader_filters_empty_parts():
    """Test that messages with empty parts are filtered."""
    export = [
        {
            "title": "Empty parts",
            "conversation_id": "conv-1",
            "mapping": {
                "msg-1": {
                    "id": "msg-1",
                    "message": {
                        "id": "msg-1",
                        "author": {"role": "user"},
                        "create_time": 1700000000.0,
                        "content": {"content_type": "text", "parts": []},
                        "metadata": {},
                    },
                    "parent": None,
                    "children": [],
                },
                "msg-2": {
                    "id": "msg-2",
                    "message": {
                        "id": "msg-2",
                        "author": {"role": "user"},
                        "create_time": 1700000010.0,
                        "content": {"content_type": "text", "parts": [""]},
                        "metadata": {},
                    },
                    "parent": None,
                    "children": [],
                },
                "msg-3": {
                    "id": "msg-3",
                    "message": {
                        "id": "msg-3",
                        "author": {"role": "user"},
                        "create_time": 1700000020.0,
                        "content": {"content_type": "text", "parts": ["   "]},
                        "metadata": {},
                    },
                    "parent": None,
                    "children": [],
                },
                "msg-4": {
                    "id": "msg-4",
                    "message": {
                        "id": "msg-4",
                        "author": {"role": "user"},
                        "create_time": 1700000030.0,
                        "content": {"content_type": "text", "parts": ["Valid"]},
                        "metadata": {},
                    },
                    "parent": None,
                    "children": [],
                },
            },
        }
    ]

    loader = ChatGPTLoader(_make_chatgpt_export(export))
    items = list(loader.load())

    assert len(items) == 1
    assert items[0].text == "Valid"


def test_chatgpt_loader_multiple_conversations():
    """Test multiple conversations are all yielded."""
    export = [
        {
            "title": "Conv 1",
            "conversation_id": "conv-1",
            "mapping": {
                "msg-1": {
                    "id": "msg-1",
                    "message": {
                        "id": "msg-1",
                        "author": {"role": "user"},
                        "create_time": 1700000000.0,
                        "content": {"content_type": "text", "parts": ["Message 1"]},
                        "metadata": {},
                    },
                    "parent": None,
                    "children": [],
                }
            },
        },
        {
            "title": "Conv 2",
            "conversation_id": "conv-2",
            "mapping": {
                "msg-2": {
                    "id": "msg-2",
                    "message": {
                        "id": "msg-2",
                        "author": {"role": "user"},
                        "create_time": 1700000000.0,
                        "content": {"content_type": "text", "parts": ["Message 2"]},
                        "metadata": {},
                    },
                    "parent": None,
                    "children": [],
                }
            },
        },
    ]

    loader = ChatGPTLoader(_make_chatgpt_export(export))
    items = list(loader.load())

    assert len(items) == 2
    assert items[0].metadata["conversation_id"] == "conv-1"
    assert items[1].metadata["conversation_id"] == "conv-2"


def test_chatgpt_loader_model_slug_included_when_present():
    """Test that model_slug is included in metadata when present."""
    export = [
        {
            "title": "Test",
            "conversation_id": "conv-1",
            "mapping": {
                "msg-1": {
                    "id": "msg-1",
                    "message": {
                        "id": "msg-1",
                        "author": {"role": "assistant"},
                        "create_time": 1700000000.0,
                        "content": {"content_type": "text", "parts": ["Response"]},
                        "metadata": {"model_slug": "gpt-4o"},
                    },
                    "parent": None,
                    "children": [],
                }
            },
        }
    ]

    loader = ChatGPTLoader(_make_chatgpt_export(export))
    items = list(loader.load())

    assert len(items) == 1
    assert items[0].metadata["model_slug"] == "gpt-4o"


def test_chatgpt_loader_omits_model_slug_when_absent():
    """Test that model_slug is omitted when not present."""
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
                        "content": {"content_type": "text", "parts": ["User message"]},
                        "metadata": {},
                    },
                    "parent": None,
                    "children": [],
                }
            },
        }
    ]

    loader = ChatGPTLoader(_make_chatgpt_export(export))
    items = list(loader.load())

    assert len(items) == 1
    assert "model_slug" not in items[0].metadata


def test_chatgpt_loader_invalid_json_raises_value_error():
    """Test that invalid JSON raises ValueError."""
    with pytest.raises(ValueError, match="invalid JSON"):
        loader = ChatGPTLoader(b"not json")
        list(loader.load())


def test_chatgpt_loader_root_not_list_raises_value_error():
    """Test that root not being a list raises ValueError."""
    with pytest.raises(ValueError, match="expected a JSON array"):
        loader = ChatGPTLoader(json.dumps({"key": "value"}).encode())
        list(loader.load())


def test_chatgpt_loader_accepts_str_input():
    """Test that constructor accepts str as well as bytes."""
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

    loader = ChatGPTLoader(json.dumps(export))  # Pass as str
    items = list(loader.load())

    assert len(items) == 1
    assert items[0].text == "Hello"


def test_chatgpt_loader_skips_nodes_without_message():
    """Test that nodes without message field are skipped silently."""
    export = [
        {
            "title": "Test",
            "conversation_id": "conv-1",
            "mapping": {
                "node-1": {
                    "id": "node-1",
                    "message": None,
                    "parent": None,
                    "children": [],
                },
                "node-2": {
                    "id": "node-2",
                    "message": {
                        "id": "node-2",
                        "author": {"role": "user"},
                        "create_time": 1700000000.0,
                        "content": {"content_type": "text", "parts": ["Valid"]},
                        "metadata": {},
                    },
                    "parent": None,
                    "children": [],
                },
            },
        }
    ]

    loader = ChatGPTLoader(_make_chatgpt_export(export))
    items = list(loader.load())

    assert len(items) == 1
    assert items[0].text == "Valid"


def test_chatgpt_loader_empty_array_yields_nothing():
    """Test that an empty export array yields no items."""
    loader = ChatGPTLoader(b"[]")
    items = list(loader.load())

    assert items == []


def test_chatgpt_loader_multiple_parts_joined():
    """Test that multiple parts are joined with spaces."""
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
                        "content": {
                            "content_type": "text",
                            "parts": ["Part 1", "Part 2", "Part 3"],
                        },
                        "metadata": {},
                    },
                    "parent": None,
                    "children": [],
                }
            },
        }
    ]

    loader = ChatGPTLoader(_make_chatgpt_export(export))
    items = list(loader.load())

    assert len(items) == 1
    assert items[0].text == "Part 1 Part 2 Part 3"

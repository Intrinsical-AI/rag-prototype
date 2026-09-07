"""Tests for POST /api/docs/import-conversations endpoint."""

import io
import json

import pytest

from local_rag_backend.core.use_cases.docs_import import ImportFileTooLargeError
from local_rag_backend.http.routers import docs as docs_router
from local_rag_backend.settings import settings


async def test_import_chatgpt_export_sparse(asgi_client, in_memory_sqlite, monkeypatch):
    """Test importing a ChatGPT export in sparse mode."""
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

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
                        "content": {"content_type": "text", "parts": ["Hi"]},
                        "metadata": {"model_slug": "gpt-4o"},
                    },
                    "parent": "msg-1",
                    "children": [],
                },
            },
        }
    ]

    files = {
        "file": (
            "export.json",
            io.BytesIO(json.dumps(export).encode()),
            "application/json",
        )
    }
    r = await asgi_client.post("/api/docs/import-conversations", files=files)

    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2
    assert len(data["ids"]) == 2
    assert data["format_detected"] == "chatgpt_export"


async def test_import_gemini_export_sparse(asgi_client, in_memory_sqlite, monkeypatch):
    """Test importing a Gemini export in sparse mode."""
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

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

    files = {
        "file": (
            "export.json",
            io.BytesIO(json.dumps(export).encode()),
            "application/json",
        )
    }
    r = await asgi_client.post("/api/docs/import-conversations", files=files)

    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2
    assert len(data["ids"]) == 2
    assert data["format_detected"] == "gemini_export"


async def test_import_unknown_format_returns_422(asgi_client, in_memory_sqlite, monkeypatch):
    """Test that unknown JSON format returns 422."""
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    unknown_export = [{"foo": "bar"}, {"baz": "qux"}]

    files = {
        "file": (
            "unknown.json",
            io.BytesIO(json.dumps(unknown_export).encode()),
            "application/json",
        )
    }
    r = await asgi_client.post("/api/docs/import-conversations", files=files)

    assert r.status_code == 422
    assert "Could not detect" in r.json()["detail"]


async def test_import_empty_file_returns_422(asgi_client, in_memory_sqlite, monkeypatch):
    """Test that empty file returns 422."""
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    files = {
        "file": (
            "empty.json",
            io.BytesIO(b""),
            "application/json",
        )
    }
    r = await asgi_client.post("/api/docs/import-conversations", files=files)

    assert r.status_code == 422
    assert "empty" in r.json()["detail"].lower()


async def test_import_invalid_json_returns_422(asgi_client, in_memory_sqlite, monkeypatch):
    """Test that invalid JSON returns 422."""
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    files = {
        "file": (
            "invalid.json",
            io.BytesIO(b"not valid json"),
            "application/json",
        )
    }
    r = await asgi_client.post("/api/docs/import-conversations", files=files)

    assert r.status_code == 422


async def test_import_empty_conversations_returns_zero_count(
    asgi_client, in_memory_sqlite, monkeypatch
):
    """Test that export with no messages returns count=0."""
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    # All messages have empty content, will be filtered out
    export = [
        {
            "title": "Empty conv",
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
                }
            },
        }
    ]

    files = {
        "file": (
            "empty.json",
            io.BytesIO(json.dumps(export).encode()),
            "application/json",
        )
    }
    r = await asgi_client.post("/api/docs/import-conversations", files=files)

    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 0
    assert data["ids"] == []
    assert data["format_detected"] == "chatgpt_export"


async def test_import_missing_file_field_returns_422(asgi_client, in_memory_sqlite, monkeypatch):
    """Test that missing 'file' field returns 422."""
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    # Send request without 'file' field
    r = await asgi_client.post("/api/docs/import-conversations")

    assert r.status_code == 422


async def test_import_file_too_large_returns_413(asgi_client, in_memory_sqlite, monkeypatch):
    """Test that file exceeding size limit returns 413."""
    from local_rag_backend.core.use_cases import docs_import as docs_import_use_case
    from local_rag_backend.http.routers import docs as docs_router

    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    def _execute_import_docs_sync_with_small_limit(**kwargs):
        return docs_import_use_case.execute_import_docs_sync(
            max_bytes=512,
            **kwargs,
        )

    monkeypatch.setattr(
        docs_router,
        "execute_import_docs_sync",
        _execute_import_docs_sync_with_small_limit,
        raising=True,
    )

    # Keep payload intentionally small to avoid heavy test runtime; limit is monkeypatched above.
    large_content = b"[" + b"x" * 2048 + b"]"

    files = {
        "file": (
            "too_large.json",
            io.BytesIO(large_content),
            "application/json",
        )
    }
    r = await asgi_client.post("/api/docs/import-conversations", files=files)

    assert r.status_code == 413
    assert "too large" in r.json()["detail"].lower()


async def test_import_chatgpt_filters_non_text(asgi_client, in_memory_sqlite, monkeypatch):
    """Test that ChatGPT import filters non-text content types."""
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

    export = [
        {
            "title": "Mixed",
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
                    "children": ["msg-2"],
                },
                "msg-2": {
                    "id": "msg-2",
                    "message": {
                        "id": "msg-2",
                        "author": {"role": "assistant"},
                        "create_time": 1700000010.0,
                        "content": {"content_type": "text", "parts": ["Response"]},
                        "metadata": {},
                    },
                    "parent": "msg-1",
                    "children": [],
                },
            },
        }
    ]

    files = {
        "file": (
            "mixed.json",
            io.BytesIO(json.dumps(export).encode()),
            "application/json",
        )
    }
    r = await asgi_client.post("/api/docs/import-conversations", files=files)

    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 1  # Only the text message
    assert data["format_detected"] == "chatgpt_export"


async def test_import_response_includes_all_fields(asgi_client, in_memory_sqlite, monkeypatch):
    """Test that response includes all required fields."""
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)

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

    files = {
        "file": (
            "export.json",
            io.BytesIO(json.dumps(export).encode()),
            "application/json",
        )
    }
    r = await asgi_client.post("/api/docs/import-conversations", files=files)

    assert r.status_code == 200
    data = r.json()
    assert "count" in data
    assert "ids" in data
    assert "format_detected" in data
    assert isinstance(data["count"], int)
    assert isinstance(data["ids"], list)
    assert isinstance(data["format_detected"], str)


@pytest.mark.unit
async def test_read_upload_with_limit_stops_early_on_oversize_chunk():
    class FakeUpload:
        def __init__(self) -> None:
            self.calls = 0

        async def read(self, _n: int = -1) -> bytes:
            self.calls += 1
            if self.calls == 1:
                # Simulate an uploader that returns more bytes than requested.
                return b"x" * 1025
            raise AssertionError("read() must not be called again after oversize detection")

    fake = FakeUpload()
    with pytest.raises(ImportFileTooLargeError):
        await docs_router._read_upload_with_limit(file=fake, max_bytes=1024, chunk_bytes=128)
    assert fake.calls == 1

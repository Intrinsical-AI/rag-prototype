"""ChatGPT conversation export loader."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from local_rag_backend.core.domain.entities import LoadedItem

if TYPE_CHECKING:
    from collections.abc import Iterable


class ChatGPTLoader:
    """
    Loader for ChatGPT conversation exports (conversations.json).

    Expects the JSON array format produced by ChatGPT's "Export data" feature.
    One LoadedItem is emitted per message, filtered by content_type == "text"
    and non-empty parts.
    """

    def __init__(self, content: bytes | str) -> None:
        """Initialize with raw JSON content (bytes or str)."""
        if isinstance(content, bytes):
            self._content = content.decode("utf-8", errors="replace")
        else:
            self._content = content

    def load(self) -> Iterable[LoadedItem]:
        """Load conversations and yield LoadedItem per message."""
        try:
            data = json.loads(self._content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"ChatGPTLoader: invalid JSON — {exc}") from exc

        if not isinstance(data, list):
            raise ValueError("ChatGPTLoader: expected a JSON array at the root level")

        for conv in data:
            if not isinstance(conv, dict):
                continue
            yield from self._load_conversation(conv)

    def _load_conversation(self, conv: dict[str, Any]) -> Iterable[LoadedItem]:
        """Extract messages from a single conversation."""
        conv_id: str = str(conv.get("conversation_id") or "")
        title: str = str(conv.get("title") or "")
        mapping: dict[str, Any] = conv.get("mapping") or {}

        for node in mapping.values():
            if not isinstance(node, dict):
                continue

            msg = node.get("message")
            if not msg or not isinstance(msg, dict):
                continue

            content_block = msg.get("content") or {}
            if not isinstance(content_block, dict):
                continue

            if content_block.get("content_type") != "text":
                continue

            parts = content_block.get("parts") or []
            text = " ".join(str(p) for p in parts if p and str(p).strip())
            if not text.strip():
                continue

            author = msg.get("author") or {}
            role: str = str(author.get("role") or "")
            msg_id: str = str(msg.get("id") or "")
            create_time = msg.get("create_time")

            # model_slug only if present
            model_slug: str | None = None
            msg_metadata = msg.get("metadata")
            if isinstance(msg_metadata, dict):
                model_slug = msg_metadata.get("model_slug")

            md: dict[str, Any] = {
                "source": "chatgpt_export",
                "conversation_id": conv_id,
                "conversation_title": title,
                "message_id": msg_id,
                "role": role,
                "created_at": create_time,
            }
            if model_slug is not None:
                md["model_slug"] = str(model_slug)

            yield LoadedItem(text=text.strip(), metadata=md)

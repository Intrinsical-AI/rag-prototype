"""Google Gemini conversation export loader (Google Takeout format)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from local_rag_backend.core.domain.entities import LoadedItem
from local_rag_backend.infrastructure.ingestion.loaders.lineage import loader_lineage

if TYPE_CHECKING:
    from collections.abc import Iterable


class GeminiLoader:
    """
    Loader for Google Gemini conversation exports (Google Takeout format).

    Expects the JSON array format produced by Google Takeout for Gemini.
    One LoadedItem is emitted per message with non-empty content.
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
            raise ValueError(f"GeminiLoader: invalid JSON — {exc}") from exc

        if not isinstance(data, list):
            raise ValueError("GeminiLoader: expected a JSON array at the root level")

        for conv in data:
            if not isinstance(conv, dict):
                continue
            yield from self._load_conversation(conv)

    def _load_conversation(self, conv: dict[str, Any]) -> Iterable[LoadedItem]:
        """Extract messages from a single conversation."""
        conv_id: str = str(conv.get("conversation_id") or "")
        title: str = str(conv.get("title") or "")
        messages = conv.get("messages") or []

        if not isinstance(messages, list):
            return

        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue

            text = str(msg.get("content") or "").strip()
            if not text:
                continue

            role: str = str(msg.get("role") or "")
            timestamp = msg.get("timestamp")

            yield LoadedItem(
                text=text,
                lineage=loader_lineage(
                    source_uri=f"gemini://conversation/{conv_id}",
                    loader_name="GeminiLoader",
                    source_version="gemini-export-v1",
                    record_locator=f"message:{idx}",
                ),
                metadata={
                    "source": "gemini_export",
                    "conversation_id": conv_id,
                    "title": title,
                    "role": role,
                    "timestamp": timestamp,
                },
            )

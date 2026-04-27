"""Core service-layer data types (transport-agnostic DTOs)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# --- CHUNKING ---
@dataclass(frozen=True)
class TextChunk:
    text: str
    chunk_index: int
    start_char: int
    end_char: int  # exclusive


# --- INFRASTRUCTURE INGESTION DETECTION ---
DetectedFormat = Literal[
    "csv", "markdown", "text", "binary", "unknown", "chatgpt_export", "gemini_export"
]


@dataclass(frozen=True)
class Detection:
    fmt: DetectedFormat
    reason: str
    mime: str | None = None

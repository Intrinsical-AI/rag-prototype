# src/local_rag_backend/infrastructure/ingestion/loaders/factory.py
"""
Factory/strategy utilities for selecting a loader for a file.

Goal:
- Don't trust extension only.
- Best-effort type detection via bytes sniffing.
- Optionally use `python-magic` (extra) when installed.
"""

from __future__ import annotations

import importlib
import math
from typing import TYPE_CHECKING, Any

from local_rag_backend.core.services.types import DetectedFormat, Detection
from local_rag_backend.infrastructure.ingestion.loaders.csv_loader import CSVLoader
from local_rag_backend.infrastructure.ingestion.loaders.markdown_loader import MarkdownLoader
from local_rag_backend.infrastructure.ingestion.loaders.text_loader import TextFileLoader

if TYPE_CHECKING:
    from pathlib import Path

    from local_rag_backend.core.ports import LoaderPort

_MIN_TEXT_RATIO = 0.95


def detect_file_format(path: Path, *, sniff_bytes: int = 4096, use_magic: bool = True) -> Detection:
    """
    Detect file kind best-effort.

    This is intentionally conservative:
    - `binary` is returned when bytes look non-textual (NUL bytes).
    - `unknown` is used when we can't confidently parse as text/csv/markdown.
    """
    ext = path.suffix.lower()
    ext_hint: DetectedFormat | None = None
    if ext in {".md", ".markdown"}:
        ext_hint = "markdown"
    elif ext == ".csv":
        ext_hint = "csv"
    elif ext in {".txt", ".log", ".rst"}:
        ext_hint = "text"

    try:
        raw = _read_head(path, sniff_bytes)
    except OSError:
        # Keep ingestion resilient: unreadable files should be skipped, not abort the full run.
        return Detection("unknown", "read-error")

    if not raw:
        return Detection("unknown", "empty")
    if b"\x00" in raw:
        return Detection("binary", "nul-byte")

    # Optional python-magic (libmagic).
    if use_magic:
        mime = _magic_mime(raw)
        if mime:
            if mime == "text/csv":
                return Detection("csv", "magic", mime=mime)
            if mime in {"text/markdown", "text/x-markdown"}:
                return Detection("markdown", "magic", mime=mime)
            if mime.startswith("text/"):
                # We'll still try to refine to csv/markdown via heuristics below.
                pass
            elif mime in {
                "application/octet-stream",
                "application/x-dosexec",
                "application/pdf",
                "image/png",
                "image/jpeg",
            }:
                return Detection("binary", "magic", mime=mime)

    # Heuristics based on decoded sample.
    text = _decode_text_sample(raw)
    if text is None:
        return Detection("unknown", "decode-failed")

    if _looks_like_markdown(text):
        return Detection("markdown", "heuristic")
    if _looks_like_csv(text):
        return Detection("csv", "heuristic")
    if _looks_like_text(text):
        if ext_hint is not None:
            return Detection(ext_hint, "extension+text")
        return Detection("text", "heuristic")

    return Detection("unknown", "no-signal")


def get_loader_for_file(
    path: Path,
    *,
    sniff_bytes: int = 4096,
    use_magic: bool = True,
    csv_delimiter: str | None = None,
    csv_has_header: bool = True,
    detection: Detection | None = None,
) -> LoaderPort | None:
    det = detection or detect_file_format(path, sniff_bytes=sniff_bytes, use_magic=use_magic)
    if det.fmt == "binary" or det.fmt == "unknown":
        return None
    if det.fmt == "csv":
        return CSVLoader(path, delimiter=csv_delimiter, has_header=csv_has_header)
    if det.fmt == "markdown":
        return MarkdownLoader(path)
    return TextFileLoader(path)


def _read_head(path: Path, n: int) -> bytes:
    with path.open("rb") as f:
        return f.read(max(0, int(n)))


def _magic_mime(raw: bytes) -> str | None:
    try:
        magic = importlib.import_module("magic")
        from_buffer = getattr(magic, "from_buffer", None)
        if from_buffer is None:
            return None
        # `magic.from_buffer` exists in python-magic; ask for MIME to keep stable signals.
        return str(from_buffer(raw, mime=True))
    except Exception:
        return None


def _decode_text_sample(raw: bytes) -> str | None:
    try:
        return raw.decode("utf-8")
    except Exception:
        try:
            # Latin-1 never fails; we'll validate printability below.
            return raw.decode("latin-1")
        except Exception:
            return None


def _looks_like_text(text: str) -> bool:
    if not text.strip():
        return False
    # Consider it text if most chars are printable/whitespace; stop early when outcome is certain.
    # Use Unicode-aware printability so UTF-8 non-ASCII texts aren't dropped as "unknown".
    total = len(text)
    required_good = math.ceil(total * _MIN_TEXT_RATIO)
    good = 0
    for i, ch in enumerate(text, 1):
        if ch.isspace() or ch.isprintable():
            good += 1
            if good >= required_good:
                return True
        remaining = total - i
        if good + remaining < required_good:
            return False
    return good >= required_good


def _looks_like_markdown(text: str) -> bool:
    t = text.lstrip()
    if t.startswith("#"):
        return True
    # Common markdown structures.
    if "```" in text:
        return True
    if "](" in text and "[" in text:
        return True
    return bool("\n- " in text or "\n* " in text)


def _looks_like_csv(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()][:5]
    if len(lines) < 2:
        return False
    candidates = [",", ";", "\t", "|"]
    for delim in candidates:
        counts = [ln.count(delim) for ln in lines]
        if max(counts) <= 0:
            continue
        # Similar delimiter counts across lines suggests tabular structure.
        if max(counts) - min(counts) <= 1 and max(counts) >= 1:
            return True
    return False


# --- JSON export detection (for UploadFile/import endpoint) ---


def _is_chatgpt_export(data: list[Any]) -> bool:
    """
    Structural heuristic: ChatGPT export has list of objects each with a 'mapping' dict key.
    We inspect at most the first 3 elements to avoid O(n) scanning of large exports.
    """
    if not data:
        return False
    samples = data[:3]
    return all(isinstance(item, dict) and "mapping" in item for item in samples)


def _is_gemini_export(data: list[Any]) -> bool:
    """
    Structural heuristic: Gemini export has list of objects each with a 'messages' list key
    whose items have 'role' and 'content' fields.
    """
    if not data:
        return False
    samples = data[:3]
    if not all(isinstance(item, dict) and "messages" in item for item in samples):
        return False

    message_lists: list[list[Any]] = []
    for item in samples:
        msgs = item.get("messages")
        if not isinstance(msgs, list):
            return False
        message_lists.append(msgs)

    # Accept minimal Gemini exports where sampled conversations may contain no messages yet.
    if all(len(msgs) == 0 for msgs in message_lists):
        return True

    # If messages exist, at least one sampled message should carry role/content.
    for msgs in message_lists:
        for msg in msgs:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                return True
    return False


def detect_json_export_format(raw: bytes) -> Detection:
    """
    Detect if a bytes payload is a ChatGPT or Gemini export JSON.

    Returns Detection with fmt in {"chatgpt_export", "gemini_export", "unknown"}.
    Does NOT call detect_file_format (that requires a Path).

    This function is intentionally conservative: only returns a non-unknown format
    when the structural heuristic is unambiguous.
    """
    import json as _json

    stripped = raw.lstrip()
    if not stripped:
        return Detection("unknown", "not-a-json-array")

    try:
        data = _json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return Detection("unknown", "json-parse-error")

    if not isinstance(data, list):
        if data is None:
            # Keep backward-compatible reason for JSON null payloads.
            return Detection("unknown", "not-a-json-array")
        return Detection("unknown", "not-a-list")

    if _is_chatgpt_export(data):
        return Detection("chatgpt_export", "structural-heuristic")
    if _is_gemini_export(data):
        return Detection("gemini_export", "structural-heuristic")

    return Detection("unknown", "json-array-unrecognized")

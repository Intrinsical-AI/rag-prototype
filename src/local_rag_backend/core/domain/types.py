"""Core domain identity and lineage types."""

from __future__ import annotations

import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, NewType

DocId = NewType("DocId", str)


@dataclass(frozen=True)
class TransformStep:
    name: str
    version: str
    params: dict[str, Any] | None = None
    timestamp: datetime | None = None


@dataclass(frozen=True)
class ItemLineage:
    source_uri: str
    loader_name: str
    source_version: str | None = None
    record_locator: str | None = None
    captured_at: datetime | None = None
    offset_start: int | None = None
    offset_end: int | None = None
    transforms: tuple[TransformStep, ...] = ()


def uuid7() -> uuid.UUID:
    """Generate an RFC4122-compatible UUIDv7 value."""
    unix_ms = int(time.time_ns() // 1_000_000)
    if unix_ms >= (1 << 48):
        raise OverflowError("unix timestamp does not fit in UUIDv7")

    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)

    uuid_int = (unix_ms << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    return uuid.UUID(int=uuid_int)


def new_doc_id(*, prefix: str = "doc") -> DocId:
    return DocId(f"{prefix}:{uuid7()}")


def utc_now() -> datetime:
    return datetime.now(UTC)

"""Shared transport schemas reused across bounded API contexts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class DocumentInDB(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    content: str
    external_id: str | None = None
    source_id: str | None = None
    metadata: dict[str, Any] | None = None

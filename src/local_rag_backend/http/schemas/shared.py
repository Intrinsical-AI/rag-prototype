"""Shared transport schemas reused across bounded API contexts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


class DocumentInDB(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    content: str
    external_id: str | None = None
    source_id: str | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("id", "content", mode="before")
    @classmethod
    def _stringify_required(cls, value: Any) -> str:
        return str(value)

    @field_validator("external_id", "source_id", mode="before")
    @classmethod
    def _stringify_optional(cls, value: Any) -> str | None:
        return None if value is None else str(value)

"""Shared transport schemas reused across bounded API contexts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DocumentInDB(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    content: str

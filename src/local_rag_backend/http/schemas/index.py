"""Index bounded-context transport schemas."""

from __future__ import annotations

from pydantic import BaseModel


class RebuildIndexResponse(BaseModel):
    indexed: int

"""Storage profiles and capability gates for mutation safety policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


class StorageCapability(StrEnum):
    ATOMIC = "ATOMIC"
    DURABLE_SAGA = "DURABLE_SAGA"
    READ_ONLY = "READ_ONLY"


@dataclass(frozen=True)
class StorageProfile:
    profile_id: str
    capabilities: frozenset[StorageCapability]
    supports_vectors: bool
    description: str

    def has(self, capability: StorageCapability) -> bool:
        return capability in self.capabilities


def default_storage_profiles() -> dict[str, StorageProfile]:
    return {
        "sql_only_local": StorageProfile(
            profile_id="sql_only_local",
            capabilities=frozenset({StorageCapability.ATOMIC, StorageCapability.DURABLE_SAGA}),
            supports_vectors=False,
            description="Single-store SQL profile for sparse mode.",
        ),
        "sql_faiss_local": StorageProfile(
            profile_id="sql_faiss_local",
            capabilities=frozenset({StorageCapability.DURABLE_SAGA}),
            supports_vectors=True,
            description="SQL + FAISS-like vector store with durable saga semantics.",
        ),
        "sql_numpy_local": StorageProfile(
            profile_id="sql_numpy_local",
            capabilities=frozenset({StorageCapability.DURABLE_SAGA}),
            supports_vectors=True,
            description="SQL + numpy vector store with durable saga semantics.",
        ),
    }


class StorageProfileRegistry:
    def __init__(self, profiles: Mapping[str, StorageProfile] | None = None) -> None:
        self._profiles = dict(profiles or default_storage_profiles())

    def get(self, profile_id: str) -> StorageProfile:
        key = str(profile_id).strip()
        if not key:
            raise ValueError("storage profile id must not be blank")
        profile = self._profiles.get(key)
        if profile is None:
            known = ", ".join(sorted(self._profiles))
            raise ValueError(f"Unknown storage profile: {key!r}. Known profiles: {known}")
        return profile

    def resolve(
        self,
        *,
        profile_id: str | None,
        retrieval_mode: str,
        vector_backend: str,
    ) -> StorageProfile:
        explicit = str(profile_id or "").strip()
        if explicit:
            return self.get(explicit)

        mode = str(retrieval_mode).strip().lower()
        if mode == "sparse":
            return self.get("sql_only_local")
        if str(vector_backend).strip().lower() == "numpy":
            return self.get("sql_numpy_local")
        return self.get("sql_faiss_local")


__all__ = [
    "StorageCapability",
    "StorageProfile",
    "StorageProfileRegistry",
    "default_storage_profiles",
]

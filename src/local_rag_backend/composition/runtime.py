"""Typed runtime snapshot derived from Settings for agent-facing surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from local_rag_backend.settings import Settings


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    """Small, explicit runtime view for status/MCP/agent flows."""

    host: str
    port: int
    debug: bool
    enable_monitoring: bool
    public_bind_requires_api_key: bool
    api_key_configured: bool
    mutation_recovery_enabled: bool
    persistence_backend: str
    search_backend: str
    retrieval_mode: str
    vector_backend: str
    storage_profile: str
    data_dir: str
    index_path: str
    id_map_path: str
    eval_dataset_path: str
    ollama_enabled: bool
    openai_enabled: bool
    openrouter_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        """Render a JSON-friendly payload for agent-facing status responses."""
        return {
            "topology": {
                "host": self.host,
                "port": self.port,
                "debug": self.debug,
                "enable_monitoring": self.enable_monitoring,
            },
            "safety": {
                "public_bind_requires_api_key": self.public_bind_requires_api_key,
                "api_key_configured": self.api_key_configured,
                "mutation_recovery_enabled": self.mutation_recovery_enabled,
            },
            "backends": {
                "persistence": self.persistence_backend,
                "search": self.search_backend,
                "retrieval": self.retrieval_mode,
                "vector": self.vector_backend,
                "storage_profile": self.storage_profile,
            },
            "paths": {
                "data_dir": self.data_dir,
                "index_path": self.index_path,
                "id_map_path": self.id_map_path,
                "eval_dataset_path": self.eval_dataset_path,
            },
            "llm": {
                "ollama_enabled": self.ollama_enabled,
                "openai_enabled": self.openai_enabled,
                "openrouter_enabled": self.openrouter_enabled,
            },
        }


def build_runtime_snapshot(settings_obj: Settings) -> RuntimeSnapshot:
    """Build a stable runtime snapshot from the mutable Settings object."""
    return RuntimeSnapshot(
        host=str(settings_obj.app_host),
        port=int(settings_obj.app_port),
        debug=bool(settings_obj.debug),
        enable_monitoring=bool(settings_obj.enable_monitoring),
        public_bind_requires_api_key=bool(settings_obj.public_bind_requires_api_key),
        api_key_configured=bool(getattr(settings_obj, "api_key", None)),
        mutation_recovery_enabled=bool(settings_obj.mutation_recovery_enabled),
        persistence_backend=str(settings_obj.persistence_backend),
        search_backend=str(settings_obj.search_backend),
        retrieval_mode=str(settings_obj.retrieval_mode),
        vector_backend=str(settings_obj.vector_backend),
        storage_profile=str(getattr(settings_obj, "storage_profile", "") or ""),
        data_dir=str(settings_obj.data_dir),
        index_path=str(settings_obj.index_path),
        id_map_path=str(settings_obj.id_map_path),
        eval_dataset_path=str(settings_obj.eval_dataset_path),
        ollama_enabled=bool(settings_obj.ollama_enabled),
        openai_enabled=bool(bool(getattr(settings_obj, "openai_api_key", None))),
        openrouter_enabled=bool(
            getattr(settings_obj, "openrouter_enabled", False)
            and getattr(settings_obj, "openrouter_api_key", None)
        ),
    )


__all__ = ["RuntimeSnapshot", "build_runtime_snapshot"]

"""Compatibility imports for legacy composition callers.

Installed consumers should use :mod:`local_rag_backend.integrations.embeddings`.
"""

from local_rag_backend.integrations.embeddings._factory import (
    DEFAULT_DENSE_BACKEND_MESSAGE as DEFAULT_DENSE_BACKEND_MESSAGE,
    _build_default_openai_embedder as _build_default_openai_embedder,
    _build_default_st_embedder as _build_default_st_embedder,
    _settings_cfg_version as _settings_cfg_version,
    build_dense_embedder_from_settings as build_dense_embedder_from_settings,
)

__all__ = ["DEFAULT_DENSE_BACKEND_MESSAGE", "build_dense_embedder_from_settings"]

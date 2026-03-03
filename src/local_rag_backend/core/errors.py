"""
Cross-layer typed errors for external provider interactions.

These errors are transport-agnostic so infrastructure adapters don't depend on FastAPI.
"""

from __future__ import annotations


class RagBaseError(Exception):
    """Base error type for RAG runtime failures."""


class EmbeddingsBackendUnavailableError(RagBaseError, RuntimeError):
    """Dense/hybrid operation needs an embeddings backend that is not available."""


class WriteLockTimeoutError(RagBaseError, RuntimeError):
    """Multi-store write lock could not be acquired before timeout."""


class LLMProviderError(RagBaseError):
    """Base error for LLM/provider integration failures."""


class LLMConfigurationError(LLMProviderError):
    """Provider is not configured correctly for the attempted operation."""


class LLMTimeoutError(LLMProviderError):
    """Provider request timed out."""


class LLMConnectionError(LLMProviderError):
    """Provider is not reachable."""


class LLMResponseError(LLMProviderError):
    """Provider returned an invalid or failed response."""

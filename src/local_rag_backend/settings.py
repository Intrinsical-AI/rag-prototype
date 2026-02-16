# src/settings.py
"""
Configuration management for Intrinsical RAG Prototype.

This module provides centralized configuration using Pydantic Settings with
environment variable support and validation. All settings can be overridden
via environment variables or .env file.

Example:
    export OPENAI_API_KEY=\"your-key-here\"
    export RETRIEVAL_MODE=\"hybrid\"
    python -m local_rag_backend.app.main
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application configuration.

    All fields have default values, so the class can be instantiated without arguments.
    """

    # This block is only for telling mypy that all fields have default values
    if False:

        def __init__(self, **kwargs: Any) -> None: ...

    # --- Core --- #
    app_host: str = Field("127.0.0.1", description="Server host IP.")
    app_port: int = Field(8000, description="Server port.")
    debug: bool = Field(False, description="Enable debug mode (auto-reload).")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", description="Logging level."
    )
    enable_monitoring: bool = Field(False, description="Enable Prometheus metrics.")

    # --- Security / HTTP --- #
    api_key: str | None = Field(
        None,
        description=(
            "Optional API key to protect HTTP endpoints. "
            "If set, clients must send it in the X-API-Key header."
        ),
    )
    public_bind_requires_api_key: bool = Field(
        True,
        description=(
            "If True, refuse to start when binding to a non-localhost address without API key. "
            "Prevents accidental exposure when using 0.0.0.0 / Docker port publishing."
        ),
    )
    cors_allow_origins: list[str] = Field(
        default_factory=list,
        description=(
            "Allowed CORS origins (exact match) when DEBUG=false. "
            "Leave empty to disable cross-origin requests."
        ),
    )

    # --- Retrieval --- #
    retrieval_mode: Literal["sparse", "dense", "hybrid"] = Field(
        # Default to sparse to keep the base installation lightweight; dense/hybrid require extra deps.
        "sparse",
        description="Retrieval strategy.",
    )
    hybrid_retrieval_alpha: float = Field(
        0.5, ge=0.0, le=1.0, description="Weight of sparse vs. dense in hybrid mode."
    )
    st_embedding_model: str = Field(
        "all-MiniLM-L6-v2", description="Sentence Transformers model for embeddings."
    )

    # --- LLM Providers --- #
    openai_api_key: str | None = Field(None, description="OpenAI API key.")
    openai_model: str = Field("gpt-4o-mini", description="Default OpenAI chat model.")
    openai_embedding_model: str = Field(
        "text-embedding-3-small", description="Default OpenAI embedding model."
    )
    openai_temperature: float = Field(0.2, ge=0.0, le=2.0, description="OpenAI temperature.")
    openai_top_p: float = Field(1.0, ge=0.0, le=1.0, description="OpenAI top_p parameter.")
    openai_max_tokens: int = Field(256, ge=1, le=4096, description="OpenAI max tokens.")
    # OpenRouter (OpenAI-compatible)
    openrouter_enabled: bool = Field(False, description="Enable OpenRouter proxy integration.")
    openrouter_api_key: str | None = Field(None, description="OpenRouter API key.")
    openrouter_base_url: str = Field(
        "https://openrouter.ai/api/v1", description="OpenRouter base URL."
    )
    openrouter_model: str = Field("openai/gpt-4o-mini", description="Default model for OpenRouter.")
    openrouter_site_url: str | None = Field(
        None, description="Public site URL for OpenRouter usage headers."
    )
    openrouter_app_title: str | None = Field(
        None, description="Application title for OpenRouter usage headers."
    )
    ollama_enabled: bool = Field(False, description="Enable Ollama integration.")
    ollama_model: str = Field("gemma3:4b", description="Default Ollama model.")
    ollama_base_url: str = Field("http://localhost:11434", description="Ollama server URL.")
    ollama_request_timeout: int = Field(180, description="Ollama request timeout in seconds.")

    # --- File Paths --- #
    data_dir: Path = Field(Path("data"), description="Base directory for data files.")
    index_path: str = Field("data/index.faiss", description="Path to the FAISS index file.")
    id_map_path: str = Field("data/id_map.json", description="Path to the FAISS ID map.")
    sqlite_url: str = Field("sqlite:///./data/app.db", description="SQLite database URL.")
    faq_csv: str = Field("data/faq.csv", description="FAQ CSV file path.")

    # --- Ingestion --- #
    ingest_chunk_strategy: Literal["chars_v1"] = Field(
        "chars_v1", description="Chunking strategy identifier (deterministic)."
    )
    ingest_chunker_version: str = Field(
        "chars_v1",
        description=(
            "Version token included in dedup hashes to force re-chunk/re-embed when changed "
            "(even if the strategy name stays the same)."
        ),
    )
    ingest_chunk_chars: int = Field(1200, ge=200, le=8000, description="Chunk size in characters.")
    ingest_chunk_overlap: int = Field(200, ge=0, le=4000, description="Overlap between chunks.")
    csv_has_header: bool = Field(True, description="Whether CSV files have header rows.")
    ingest_clean_lowercase: bool = Field(True, description="Lowercase during ingestion cleaning.")
    ingest_clean_remove_html: bool = Field(True, description="Remove HTML tags during cleaning.")
    ingest_clean_collapse_whitespace: bool = Field(
        True, description="Collapse whitespace during cleaning."
    )
    ingest_clean_strip: bool = Field(True, description="Strip leading/trailing whitespace first.")

    # --- Prompt Templates --- #
    openai_prompt_template: str = Field(
        "Answer using ONLY the context provided.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}"
    )
    ollama_prompt_template: str = Field(
        "Based on the context, answer the question.\nIf the context is not enough, say so.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Allow non-JSON env vars for complex fields (e.g., comma-separated CORS origins).
        enable_decoding=False,
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, v: Any) -> Any:
        # Make env/config more forgiving while keeping a strict Literal type.
        if isinstance(v, str):
            return v.upper()
        return v

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_cors_allow_origins(cls, v: Any) -> Any:
        """
        Allow `CORS_ALLOW_ORIGINS` to be set as:
        - JSON list (recommended): ["http://localhost:5173", ...]
        - Comma-separated string: http://localhost:5173,http://127.0.0.1:5173
        - Empty string: (disable CORS)
        """
        if v is None:
            return []
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            # JSON list string (common in docker-compose env)
            if s.startswith("["):
                try:
                    parsed = json.loads(s)
                except Exception:
                    # Fall back to comma-separated parsing.
                    parsed = None
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            return [item.strip() for item in s.split(",") if item.strip()]
        return v

    @field_validator("data_dir", mode="before")
    @classmethod
    def _normalize_data_dir(cls, v: Any) -> Path:
        # Keep Settings side-effect free; callers are responsible for creating directories.
        return Path(v)

    @field_validator("sqlite_url")
    @classmethod
    def _validate_sqlite_url(cls, v: str) -> str:
        if not v.startswith("sqlite:///"):
            raise ValueError("SQLite URL must start with 'sqlite:///'")
        return v

    @field_validator("ollama_base_url")
    @classmethod
    def _validate_ollama_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("Ollama URL must start with http:// or https://")
        return v.rstrip("/")

    @model_validator(mode="after")
    def _validate_chunking(self) -> Settings:
        """Ensure chunk overlap is strictly less than chunk size."""
        if self.ingest_chunk_overlap >= self.ingest_chunk_chars:
            raise ValueError("ingest_chunk_overlap must be strictly less than ingest_chunk_chars")
        return self

    def get_database_path(self) -> Path:
        """Get the database file path."""
        if self.sqlite_url.startswith("sqlite:///"):
            db_path = self.sqlite_url[10:]  # Remove 'sqlite:///'
            return Path(db_path)
        raise ValueError("Invalid SQLite URL format")


# Global settings instance
settings: Settings = Settings()

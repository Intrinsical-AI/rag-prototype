"""
Intrinsical-AI RAG Prototype
Copyright (c) 2025 Intrinsical-AI

Module: Application Settings
Purpose: Centralized configuration management using Pydantic Settings.
         Provides environment variable support, validation, and type safety.

Example:
    export OPENAI_API_KEY="your-key-here"
    export RETRIEVAL_MODE="hybrid"
    python -m local_rag_backend.app.main
"""

from __future__ import annotations

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

    # --- Retrieval --- #
    retrieval_mode: Literal["sparse", "dense", "hybrid"] = Field(
        "hybrid", description="Retrieval strategy."
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
    ollama_model: str = Field("gemma3:1b", description="Default Ollama model.")
    ollama_base_url: str = Field("http://localhost:11434", description="Ollama server URL.")
    ollama_request_timeout: int = Field(180, description="Ollama request timeout in seconds.")

    # --- File Paths --- #
    data_dir: Path = Field(Path("data"), description="Base directory for data files.")
    index_path: str = Field("data/index.faiss", description="Path to the FAISS index file.")
    id_map_path: str = Field("data/id_map.pkl", description="Path to the FAISS ID map.")
    sqlite_url: str = Field("sqlite:///./data/app.db", description="SQLite database URL.")
    faq_csv: str = Field("data/faq.csv", description="FAQ CSV file path.")

    # --- Ingestion --- #
    ingest_chunk_chars: int = Field(1200, ge=200, le=8000, description="Chunk size in characters.")
    ingest_chunk_overlap: int = Field(200, ge=0, le=4000, description="Overlap between chunks.")
    csv_has_header: bool = Field(True, description="Whether CSV files have header rows.")

    # --- Prompt Templates --- #
    openai_prompt_template: str = Field(
        "Answer using ONLY the context provided.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}"
    )
    ollama_prompt_template: str = Field(
        "Based on the context, answer the question.\nIf the context is not enough, say so.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}"
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    @field_validator("data_dir", mode="before")
    @classmethod
    def _ensure_data_dir_exists(cls, v: Any) -> Path:
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

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

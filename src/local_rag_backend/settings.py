"""
Production-ready configuration management for Intrinsical RAG Prototype.

This module provides centralized configuration using Pydantic Settings with
environment variable support and validation. All settings can be overridden
via environment variables or .env file.

Example:
    export OPENAI_API_KEY="your-key-here"
    export RETRIEVAL_MODE="hybrid"
    python -m local_rag_backend.app.main
"""

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # === APPLICATION RUNTIME ===
    app_host: str = Field(default="0.0.0.0", description="Host IP for FastAPI server")  # nosec B104, noqa: S104
    app_port: int = Field(default=8000, ge=1, le=65535, description="Port for FastAPI server")
    debug: bool = Field(default=False, description="Enable debug mode")

    # === RETRIEVAL CONFIGURATION ===
    retrieval_mode: Literal["sparse", "dense", "hybrid"] = Field(
        default="sparse", description="Retrieval mode: sparse (BM25), dense (FAISS), or hybrid"
    )

    # === OPENAI CONFIGURATION ===
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI chat model")
    openai_temperature: float = Field(
        default=0.2, ge=0.0, le=2.0, description="Sampling temperature"
    )
    openai_top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling")
    openai_max_tokens: int = Field(default=256, ge=1, le=4096, description="Max tokens in response")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )

    # === OLLAMA CONFIGURATION ===
    ollama_enabled: bool = Field(default=False, description="Enable Ollama integration")
    ollama_model: str = Field(default="gemma3:1b", description="Ollama model name")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    ollama_request_timeout: int = Field(
        default=90, ge=1, le=300, description="Request timeout in seconds"
    )

    # === SENTENCE TRANSFORMERS ===
    st_embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence Transformers model"
    )

    # === PROMPT TEMPLATES ===
    ollama_prompt_template: str = Field(
        default="Based on the following context, please answer the question.\nIf the context does not provide an answer, say so.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}",
        description="Prompt template for Ollama (use {context} and {question} placeholders)",
    )
    openai_prompt_template: str = Field(
        default="Answer using ONLY the context provided.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}",
        description="Prompt template for OpenAI (use {context} and {question} placeholders)",
    )
    hybrid_retrieval_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for sparse component in hybrid retrieval (0.0=dense only, 1.0=sparse only)",
    )

    # === FILE PATHS ===
    data_dir: Path = Field(default=Path("data"), description="Data directory path")
    index_path: str = Field(default="data/index.faiss", description="FAISS index file path")
    id_map_path: str = Field(default="data/id_map.pkl", description="FAISS ID map file path")
    faq_csv: str = Field(default="data/faq.csv", description="FAQ CSV file path")
    sqlite_url: str = Field(default="sqlite:///./data/app.db", description="SQLite database URL")

    # === DATA PROCESSING ===
    csv_has_header: bool = Field(default=True, description="CSV file has header row")
    enable_faiss_consistency_check: bool = Field(
        default=True,
        description="Enable FAISS/SQL consistency checks on startup (can be expensive for large datasets)",
    )

    # === INGESTION SETTINGS ===
    ingest_chunk_chars: int = Field(
        default=1200, ge=200, le=8000, description="Maximum characters per chunk"
    )
    ingest_chunk_overlap: int = Field(
        default=200, ge=0, le=4000, description="Character overlap between chunks"
    )

    # === LOGGING ===
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )

    # === MONITORING ===
    enable_monitoring: bool = Field(
        default=False, description="Enable Prometheus metrics collection"
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    @field_validator("data_dir", mode="before")
    @classmethod
    def ensure_data_dir_exists(cls, v: Any) -> Path:
        """Ensure data directory exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @field_validator("sqlite_url")
    @classmethod
    def validate_sqlite_url(cls, v: str) -> str:
        """Validate SQLite URL format."""
        if not v.startswith("sqlite:///"):
            raise ValueError("SQLite URL must start with 'sqlite:///'")
        return v

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_url(cls, v: str) -> str:
        """Validate Ollama URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Ollama URL must start with http:// or https://")
        return v.rstrip("/")

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"

    def get_database_path(self) -> Path:
        """Get the database file path."""
        if self.sqlite_url.startswith("sqlite:///"):
            db_path = self.sqlite_url[10:]  # Remove 'sqlite:///'
            return Path(db_path)
        raise ValueError("Invalid SQLite URL format")


# Global settings instance
settings = Settings()

# src/settings.py
"""
Configuration management for Intrinsical RAG Prototype.

This module provides centralized configuration using a YAML file as the single
runtime source of truth. The YAML payload is loaded at startup and validated
through a Pydantic model.
"""

from __future__ import annotations

import json
import os
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

DEFAULT_CONFIG_PATH = Path("config.yaml")


def _resolve_relative_path_value(value: Any, *, base_dir: Path) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    path = (base_dir / path).resolve() if not path.is_absolute() else path.resolve()
    return str(path)


def _resolve_sqlite_url_value(value: Any, *, base_dir: Path) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    prefix = "sqlite:///"
    if not rendered.startswith(prefix):
        return rendered
    raw_path = rendered[len(prefix) :]
    if not raw_path:
        return rendered
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return f"{prefix}{path}"
    resolved = (base_dir / path).resolve()
    return f"{prefix}{resolved}"


def load_settings_from_yaml(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Settings:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        data: dict[str, Any] = {}
    elif isinstance(raw, dict):
        data = dict(raw)
    else:
        raise ValueError("Configuration file must contain a YAML mapping at the top level.")

    base_dir = path.resolve().parent
    # Benchmark/orchestration runners can redirect perf metrics without editing the
    # repository config.yaml that remains the main runtime source of truth.
    perf_metrics_override = str(os.getenv("RAG_PERF_METRICS_OUT", "") or "").strip()
    if perf_metrics_override:
        data["perf_metrics_out_path"] = perf_metrics_override
    for key in (
        "data_dir",
        "index_path",
        "id_map_path",
        "faq_csv",
        "embedding_cache_db_path",
        "eval_dataset_path",
        "perf_metrics_out_path",
        "lock_metrics_path",
    ):
        if key in data:
            data[key] = _resolve_relative_path_value(data.get(key), base_dir=base_dir)
    if "sqlite_url" in data:
        data["sqlite_url"] = _resolve_sqlite_url_value(data.get("sqlite_url"), base_dir=base_dir)
    return Settings(**data)


class Settings(BaseModel):
    """Centralized application configuration.

    All fields have default values, so the class can be instantiated without arguments.
    """

    # Block for telling mypy that all fields have default values
    if False:

        def __init__(self, **kwargs: Any) -> None: ...

    persistence_backend: Literal["local_split", "elasticsearch"] = Field(
        "local_split",
        description=(
            "Persistence backend topology. "
            "'local_split' uses SQLite + local vector index; "
            "'elasticsearch' uses Elasticsearch as unified storage."
        ),
    )

    # --- Core --- #
    app_host: str = Field("127.0.0.1", description="Server host IP.")
    app_port: int = Field(8000, description="Server port.")
    debug: bool = Field(False, description="Enable debug mode (auto-reload).")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", description="Logging level."
    )
    enable_monitoring: bool = Field(False, description="Enable Prometheus metrics.")
    enable_reranker: bool = Field(
        False, description="Enable reranking of retrieved documents (best-effort)."
    )

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
    search_backend: Literal["local_split", "elasticsearch", "opensearch", "solr"] = Field(
        "local_split",
        description=(
            "Search execution backend. "
            "'local_split' queries the local SQL/vector stores; "
            "'elasticsearch' and 'opensearch' query remote search clusters; "
            "'solr' queries a remote Solr core."
        ),
    )
    retrieval_mode: Literal["sparse", "dense", "dual", "hybrid"] = Field(
        # Default to sparse to keep the base installation lightweight; dense/hybrid require extra deps.
        "sparse",
        description="Retrieval strategy.",
    )
    vector_backend: Literal["auto", "faiss", "numpy"] = Field(
        "auto",
        description=(
            "Vector index engine selection for dense/hybrid modes. "
            "'auto' prefers faiss when installed, else numpy."
        ),
    )
    hybrid_retrieval_alpha: float = Field(
        0.5, ge=0.0, le=1.0, description="Weight of sparse vs. dense in hybrid mode."
    )
    dual_candidate_k: int = Field(
        50,
        ge=1,
        le=1000,
        description="Sparse candidate count for dual retrieval before dense rerank.",
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
    openai_request_timeout: int = Field(
        60,
        ge=1,
        le=600,
        description="Timeout in seconds for OpenAI-compatible HTTP requests.",
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
    ollama_model: str = Field("lfm2.5-thinking", description="Default Ollama model.")
    ollama_base_url: str = Field("http://localhost:11434", description="Ollama server URL.")
    ollama_request_timeout: int = Field(180, description="Ollama request timeout in seconds.")

    # --- File Paths --- #
    data_dir: Path = Field(Path("data"), description="Base directory for data files.")
    index_path: str = Field("data/index.faiss", description="Path to the vector index file.")
    id_map_path: str = Field("data/id_map.json", description="Path to the vector index ID map.")
    sqlite_url: str = Field("sqlite:///./data/app.db", description="SQLite database URL.")
    faq_csv: str = Field("data/faq.csv", description="FAQ CSV file path.")
    storage_profile: str = Field(
        "",
        description=(
            "Storage profile identifier (optional). If empty, it is inferred from retrieval_mode "
            "and vector backend."
        ),
    )
    eval_dataset_path: str = Field(
        "datasets/rag_eval_v1.jsonl",
        description="Default JSONL dataset path for offline retrieval evaluation.",
    )
    embedding_cache_db_path: str = Field(
        "data/embedding_cache.sqlite3",
        description="Persistent embedding cache DB path.",
    )
    disable_embedding_cache: bool = Field(
        False,
        description="Disable the content-addressed embedding cache wrapper.",
    )
    synthetic_embeddings: bool = Field(
        False,
        description="Enable synthetic SentenceTransformer embeddings for local stress testing.",
    )
    synthetic_embedding_fail_rate: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Probability of synthetic embedding failure when synthetic_embeddings=true.",
    )
    synthetic_embedding_jitter_min_ms: float = Field(
        0.0,
        ge=0.0,
        description="Minimum synthetic embedding jitter in milliseconds.",
    )
    synthetic_embedding_jitter_max_ms: float = Field(
        0.0,
        ge=0.0,
        description="Maximum synthetic embedding jitter in milliseconds.",
    )
    synthetic_embedding_dim: int = Field(
        384,
        ge=1,
        description="Embedding dimensionality used when synthetic embeddings are enabled.",
    )
    perf_metrics_out_path: str | None = Field(
        None,
        description="Optional JSON output file for process-local perf metrics.",
    )
    lock_metrics_path: str | None = Field(
        None,
        description="Optional NDJSON output file for write-lock metrics.",
    )
    blocking_workers: int = Field(
        8,
        ge=1,
        description="Default worker count for blocking tasks.",
    )
    blocking_workers_mutation: int = Field(
        2,
        ge=1,
        description="Worker count for blocking mutation tasks.",
    )
    blocking_workers_network: int = Field(
        4,
        ge=1,
        description="Worker count for blocking network tasks.",
    )
    blocking_workers_eval: int = Field(
        2,
        ge=1,
        description="Worker count for blocking eval tasks.",
    )
    blocking_queue_default: int = Field(
        64,
        ge=1,
        description="Queue depth for default blocking tasks beyond worker count.",
    )
    blocking_queue_mutation: int = Field(
        32,
        ge=1,
        description="Queue depth for mutation blocking tasks beyond worker count.",
    )
    blocking_queue_network: int = Field(
        64,
        ge=1,
        description="Queue depth for network blocking tasks beyond worker count.",
    )
    blocking_queue_eval: int = Field(
        32,
        ge=1,
        description="Queue depth for eval blocking tasks beyond worker count.",
    )
    write_lock_timeout_s: float = Field(
        30.0,
        ge=0.1,
        le=600.0,
        description="Timeout in seconds when waiting for multi-store write lock acquisition.",
    )
    write_lock_poll_s: float = Field(
        0.05,
        ge=0.005,
        le=5.0,
        description="Polling interval in seconds for lock acquisition retries.",
    )
    mutation_batch_max_size: int = Field(
        32,
        ge=1,
        le=512,
        description=(
            "Maximum number of queued mutation requests drained by one lock holder in a single "
            "batch cycle."
        ),
    )
    mutation_batch_max_wait_ms: int = Field(
        50,
        ge=0,
        le=5000,
        description=(
            "Maximum wait (milliseconds) to coalesce additional mutation requests before "
            "draining a batch."
        ),
    )
    mutation_recovery_enabled: bool = Field(
        True,
        description="Enable startup recovery of incomplete durable mutation records.",
    )
    mutation_recovery_interval_s: float = Field(
        30.0,
        ge=1.0,
        le=3600.0,
        description="Background interval (seconds) for retrying incomplete mutation recovery.",
    )
    es_base_url: str | None = Field(None, description="Elasticsearch base URL.")
    es_api_key: str | None = Field(None, description="Elasticsearch API key.")
    es_username: str | None = Field(None, description="Elasticsearch username.")
    es_password: str | None = Field(None, description="Elasticsearch password.")
    es_verify_tls: bool = Field(True, description="Verify Elasticsearch TLS certificates.")
    es_request_timeout_s: float = Field(
        30.0, ge=0.5, le=600.0, description="Elasticsearch request timeout in seconds."
    )
    es_docs_index: str = Field("rag-docs", description="Elasticsearch index for documents.")
    es_history_index: str = Field("rag-history", description="Elasticsearch index for history.")
    es_system_index: str = Field("rag-system", description="Elasticsearch index for system state.")
    es_tombstones_index: str = Field(
        "rag-tombstones", description="Elasticsearch index for document tombstones."
    )
    es_content_field: str = Field("content", description="Elasticsearch content field name.")
    es_embedding_field: str = Field(
        "embedding", description="Elasticsearch dense vector field name."
    )
    es_hybrid_lexical_k: int = Field(
        50, ge=1, le=1000, description="Lexical candidate count for Elasticsearch hybrid."
    )
    es_hybrid_vector_k: int = Field(
        50, ge=1, le=1000, description="Vector candidate count for Elasticsearch hybrid."
    )
    os_base_url: str | None = Field(None, description="OpenSearch base URL.")
    os_api_key: str | None = Field(None, description="OpenSearch API key.")
    os_username: str | None = Field(None, description="OpenSearch username.")
    os_password: str | None = Field(None, description="OpenSearch password.")
    os_verify_tls: bool = Field(True, description="Verify OpenSearch TLS certificates.")
    os_request_timeout_s: float = Field(
        30.0, ge=0.5, le=600.0, description="OpenSearch request timeout in seconds."
    )
    os_docs_index: str = Field("rag-docs", description="OpenSearch index for documents.")
    os_content_field: str = Field("content", description="OpenSearch content field name.")
    os_embedding_field: str = Field("embedding", description="OpenSearch dense vector field name.")
    os_dense_candidate_k: int = Field(
        50, ge=1, le=1000, description="Vector candidate count for OpenSearch dense retrieval."
    )
    solr_base_url: str | None = Field(None, description="Solr base URL.")
    solr_core: str = Field("rag-docs", description="Solr core/collection for documents.")
    solr_content_field: str = Field("content", description="Solr content field name.")
    solr_request_timeout_s: float = Field(
        30.0, ge=0.5, le=600.0, description="Solr request timeout in seconds."
    )

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
    ingest_batch_size: int = Field(
        64, ge=1, le=512, description="Number of file-plans processed per ingestion batch."
    )
    ingest_clean_lowercase: bool = Field(True, description="Lowercase during ingestion cleaning.")
    ingest_clean_remove_html: bool = Field(True, description="Remove HTML tags during cleaning.")
    ingest_clean_collapse_whitespace: bool = Field(
        True, description="Collapse whitespace during cleaning."
    )
    ingest_clean_strip: bool = Field(True, description="Strip leading/trailing whitespace first.")

    # --- Retrieval quality (optional) --- #
    reranker_strategy: Literal["overlap_v1"] = Field(
        "overlap_v1", description="Reranker strategy identifier."
    )
    reranker_candidate_k: int = Field(
        20,
        ge=3,
        le=200,
        description="Candidates to fetch before reranking (top-k is returned).",
    )

    # --- Prompt Templates --- #
    openai_prompt_template: str = Field(
        "Answer using ONLY the context provided.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}"
    )
    ollama_prompt_template: str = Field(
        "Based on the context, answer the question.\nIf the context is not enough, say so.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}"
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("debug", mode="before")
    @classmethod
    def _normalize_debug_bool(cls, v: Any) -> Any:
        if isinstance(v, str):
            normalized = v.strip().lower()
            if normalized in {"debug", "development", "dev"}:
                return True
            if normalized in {"release", "production", "prod"}:
                return False
        return v

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, v: Any) -> Any:
        # Keep the YAML config forgiving while retaining a strict Literal type.
        if isinstance(v, str):
            return v.upper()
        return v

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_cors_allow_origins(cls, v: Any) -> Any:
        """
        Allow `cors_allow_origins` to be set as:
        - YAML list (recommended): ["http://localhost:5173", ...]
        - Comma-separated string: http://localhost:5173,http://127.0.0.1:5173
        - Empty string: (disable CORS)
        """
        if v is None:
            return []
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            # JSON list string (useful when the YAML loader gets a quoted scalar).
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
    def _validate_sqlite_url(cls, v: str, info: ValidationInfo) -> str:
        persistence_backend = str(info.data.get("persistence_backend") or "local_split")
        if persistence_backend == "elasticsearch":
            return v
        if not v.startswith("sqlite:///"):
            raise ValueError("SQLite URL must start with 'sqlite:///'")
        return v

    @field_validator("ollama_base_url")
    @classmethod
    def _validate_ollama_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("Ollama URL must start with http:// or https://")
        return v.rstrip("/")

    @field_validator("es_base_url")
    @classmethod
    def _validate_es_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.startswith(("http://", "https://")):
            raise ValueError("Elasticsearch URL must start with http:// or https://")
        return v.rstrip("/")

    @field_validator("os_base_url")
    @classmethod
    def _validate_os_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.startswith(("http://", "https://")):
            raise ValueError("OpenSearch URL must start with http:// or https://")
        return v.rstrip("/")

    @field_validator("solr_base_url")
    @classmethod
    def _validate_solr_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.startswith(("http://", "https://")):
            raise ValueError("Solr URL must start with http:// or https://")
        return v.rstrip("/")

    @model_validator(mode="after")
    def _validate_chunking(self) -> Settings:
        """Ensure chunk overlap is strictly less than chunk size."""
        if self.ingest_chunk_overlap >= self.ingest_chunk_chars:
            raise ValueError("ingest_chunk_overlap must be strictly less than ingest_chunk_chars")
        if self.persistence_backend == "elasticsearch":
            if self.retrieval_mode == "sparse" and self.search_backend != "elasticsearch":
                raise ValueError(
                    "persistence_backend=elasticsearch supports retrieval_mode=sparse only when "
                    "search_backend=elasticsearch"
                )
            if not self.es_base_url:
                raise ValueError("es_base_url is required when persistence_backend=elasticsearch")
        else:
            if not self.sqlite_url.startswith("sqlite:///"):
                raise ValueError("SQLite URL must start with 'sqlite:///'")

        if self.search_backend == "elasticsearch" and not self.es_base_url:
            raise ValueError("es_base_url is required when search_backend=elasticsearch")
        if self.search_backend == "opensearch" and not self.os_base_url:
            raise ValueError("os_base_url is required when search_backend=opensearch")
        if self.search_backend == "solr" and not self.solr_base_url:
            raise ValueError("solr_base_url is required when search_backend=solr")
        if self.search_backend == "solr" and self.retrieval_mode in {"dense", "dual"}:
            raise ValueError("search_backend=solr supports only retrieval_mode=sparse in v1")
        if self.retrieval_mode == "hybrid" and self.search_backend not in {
            "local_split",
            "elasticsearch",
        }:
            raise ValueError(
                "retrieval_mode=hybrid is supported only with search_backend=local_split|elasticsearch"
            )
        if (
            self.retrieval_mode == "hybrid"
            and self.search_backend == "elasticsearch"
            and self.persistence_backend != "elasticsearch"
        ):
            raise ValueError(
                "retrieval_mode=hybrid with search_backend=elasticsearch requires "
                "persistence_backend=elasticsearch"
            )
        return self

    def get_database_path(self) -> Path:
        """Get the database file path."""
        if self.sqlite_url.startswith("sqlite:///"):
            db_path = self.sqlite_url[10:]  # Remove 'sqlite:///'
            return Path(db_path)
        raise ValueError("Invalid SQLite URL format")

    def get_coordination_dir(self) -> Path:
        """
        Return the directory used for cross-process coordination artifacts.

        Priority:
        - Explicit absolute `data_dir` (user intent).
        - Parent dir of absolute SQLite path (keeps workers aligned on shared DB).
        - Resolved `data_dir` for purely relative deployments.

        Rationale:
        - A relative `data_dir` can resolve differently per process (different CWD),
          splitting write locks while sharing the same absolute SQLite database.
          Prefer the DB parent in that case to avoid multi-process lock drift.
        """
        data_dir = Path(self.data_dir).expanduser()
        if data_dir.is_absolute():
            return data_dir.resolve()
        with suppress(Exception):
            db_path = self.get_database_path().expanduser()
            if db_path.is_absolute():
                return db_path.parent.resolve()
        return data_dir.resolve()


# Global settings instance
settings: Settings = load_settings_from_yaml()

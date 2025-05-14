"""
File: src/settings.py
Path: src/settings.py
Global configuration loaded via environment variables.
Use `python-dotenv` or export vars before running.

Why? Centralises all tunables and keeps secrets out of code.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # RUNTIME
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    # RETRIEVAL
    retrieval_mode: str = Field("sparse", pattern="^(sparse|dense)$")
    # OPENAI
    openai_api_key: str | None = None
    openai_model: str = "gpt-3.5-turbo"
    # OPENAI sampling
    openai_temperature: float = 0.2
    openai_top_p: float = 1.0
    openai_max_tokens: int = 256
    openai_embedding_model: str = "text-embedding-3-small" # embeddings
    # OLLAMA
    ollama_enabled: bool = True
    ollama_model: str = "deepseek-r1:1.5B"
    ollama_base_url: str = "http://localhost:11434"
    ollama_request_timeout: int = 90 # Timeout en segundos
    # PATHS
    index_path: str = "data/index.faiss"
    id_map_path: str = "data/id_map.pkl"
    faq_csv: str = "data/faq.csv"     # for build_index.py
    sqlite_url: str = "sqlite:///./data/app.db"
    csv_has_header:bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()

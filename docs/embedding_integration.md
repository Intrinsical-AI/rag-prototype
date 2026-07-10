# Installed Embedding Integration API

Installed consumers must use `local_rag_backend.integrations.embeddings`; modules under
`cli_commands`, `composition`, and `infrastructure` are implementation details.

The package is PEP 561 typed through `local_rag_backend/py.typed`. Its supported exports are:

```python
def create_embedding_service(
    config_path: str | Path | None = None,
    *,
    limits: EmbeddingLimits | None = None,
) -> EmbeddingService: ...

class EmbeddingService(Protocol):
    def status(self) -> EmbeddingStatus: ...
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
```

`EmbeddingLimits`, `EmbeddingStatus`, `DEFAULT_EMBEDDING_LIMITS`, and
`EmbeddingsBackendUnavailableError` are also public.

## Configuration and lifecycle

Create one service and reuse it so provider initialization and the content-addressed cache remain
effective:

```python
from local_rag_backend.integrations.embeddings import create_embedding_service

service = create_embedding_service("/etc/my-app/rag.yaml")
print(service.status())
vectors = service.embed(["first text", "second text"])
```

Configuration precedence is:

1. `config_path` passed to `create_embedding_service`;
2. `RAG_CONFIG_PATH` when the argument is omitted;
3. `config.yaml` in the process working directory.

Relative data and cache paths remain rooted at the selected YAML file. Provider selection is
unchanged: a configured `openai_api_key` selects OpenAI; otherwise SentenceTransformers is used.
The configured content-addressed SQLite cache remains active unless
`disable_embedding_cache: true`. Synthetic embeddings follow the same SentenceTransformer/cache
path and need no heavy model or network dependency.

## Status and validation

`status()` contains only JSON-safe values and no credentials:

```json
{
  "provider": "sentence_transformers",
  "model": "all-MiniLM-L6-v2",
  "model_key": "sentence_transformers:all-MiniLM-L6-v2:384",
  "dimension": 384,
  "synthetic": false,
  "cache_enabled": true,
  "cache_db_path": "/absolute/path/embedding_cache.sqlite3",
  "limits": {
    "max_batch_size": 128,
    "max_text_chars": 32768,
    "max_total_chars": 262144
  }
}
```

`embed()` preserves input order and returns plain Python floats. It rejects strings passed in place
of a sequence, non-string or blank items, oversized items/batches, non-finite vectors, wrong vector
counts, and dimension drift. Input validation raises `ValueError`; malformed provider output raises
`RuntimeError`. If no configured embedding backend is available, construction raises the existing
`EmbeddingsBackendUnavailableError` contract.

Release validation must also run the installed-artifact gate:

```bash
make smoke-embedding-api-wheel
```

It builds the wheel, installs runtime dependencies into a fresh temporary environment, executes
both explicit-path and `RAG_CONFIG_PATH` flows outside the checkout, and verifies the PEP 561 marker,
synthetic cache/status/vector behavior, and absence of the heavy `sentence-transformers` package.

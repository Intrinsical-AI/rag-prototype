# Intrinsical RAG Prototype

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/Intrinsical-AI/rag-prototype/actions)
[![Coverage](https://img.shields.io/badge/coverage-85%25%2B-green.svg)](https://github.com/Intrinsical-AI/rag-prototype)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/r/intrinsical/rag-prototype)
[![PyPI](https://img.shields.io/pypi/v/rag-prototype.svg)](https://pypi.org/project/rag-prototype/)
[![Downloads](https://img.shields.io/pypi/dm/rag-prototype.svg)](https://pypi.org/project/rag-prototype/)

> General-purpose RAG system with a hexagonal architecture (Ports & Adapters), FastAPI, three retrieval modes (BM25, FAISS, hybrid), and swappable LLM connectors (OpenAI or Ollama). Designed as a solid base to iterate in real development environments.

---

## Features

* **Clean architecture**

  * Hexagonal (Ports & Adapters): domain decoupled from infrastructure.
  * Explicit typing and domain models.
* **Retrieval**

  * Sparse: BM25 (offline).
  * Dense: FAISS + SentenceTransformers.
  * Hybrid: dense + BM25 combination with configurable weight.
* **LLMs**

  * OpenAI Chat (via API key).
  * Local Ollama (over HTTP). Current clients are synchronous.
* **Persistence**

  * SQLite via SQLAlchemy: documents and Q\&A history.
  * FAISS on disk for dense/hybrid mode.
* **API**

  * FastAPI with validation and OpenAPI at `/docs`.
  * Health: `/api/health`, Readiness: `/api/ready`, Ollama health: `/api/health/ollama`.
  * Config: `/api/config`, Templates: `/api/templates`.
  * OpenRouter proxy (OpenAI-compatible): `POST /api/openrouter/generate`.
* **Tests**

  * Unit, integration, and E2E with `pytest`.

---

## Project structure

```bash
.
├── data/                      # CSV, SQLite DB, FAISS files
├── src/local_rag_backend/
│   ├── app/                   # FastAPI (main, routers, DI, factory)
│   ├── core/                  # domain, ports and services (ETL, RAG)
│   ├── infrastructure/        # adapters: llms, retrievers, storage, loaders
│   ├── scripts/               # bootstrap and build_index
│   └── frontend/              # packaged index.html to serve at /
└── tests/                     # unit + integration + e2e
```

---

## Docs

* `docs/architecture.md`
* `docs/custom_usage_guide.md`
* `docs/langchain_loaders.md`

---

## Requirements

* Python 3.11+
* Operating system: Linux / macOS / Windows
* For dense/hybrid mode: install the `dense` extra (FAISS). For SentenceTransformers embeddings, also install `dense-st`.

---

## Installation and startup (from source)

```bash
git clone https://github.com/Intrinsical-AI/rag-prototype.git
cd rag-prototype

# Recommended: uv-managed local venv + lockfile installs
# If your environment has a non-writable home directory, keep uv cache local:
# export UV_CACHE_DIR=.uv-cache
uv venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

# Install runtime deps (uses uv.lock)
uv sync --frozen

# (Optional) Dense/Hybrid deps (FAISS)
# uv sync --frozen --extra dense
#
# (Optional) SentenceTransformers embeddings (heavy: torch/transformers)
# uv sync --frozen --extra dense-st

# (Optional) Dev/Test deps
# uv sync --frozen --extra dev --extra test
```

Initialize sample data and start:

```bash
# Load sample CSV into SQLite and, if applicable, build FAISS
rag-bootstrap


# FastAPI server
rag-server
# UI: http://localhost:8000/
# Health: http://localhost:8000/api/health
# Ollama health: http://localhost:8000/api/health/ollama
# Docs: http://localhost:8000/docs
```

> If you prefer to invoke scripts directly:
> `python scripts/bootstrap.py` (repo) or `rag-bootstrap` (installed), and
> `uvicorn local_rag_backend.app.main:app --reload`.

---

## Configuration

Source of truth: `src/local_rag_backend/settings.py` (Pydantic Settings). Settings can be overridden with
environment variables or a `.env` file (case-insensitive).

Key variables (non-exhaustive):

| Variable                         | Default                   | Scope        | Description                                            |
| -------------------------------- | ------------------------- | ------------ | ------------------------------------------------------ |
| `APP_HOST`                       | `127.0.0.1`               | server       | Service host                                           |
| `APP_PORT`                       | `8000`                    | server       | Service port                                           |
| `DEBUG`                          | `false`                   | server       | Reload/detailed logging                                |
| `LOG_LEVEL`                      | `INFO`                    | server       | Logging level                                          |
| `API_KEY`                        | —                         | security     | If set, require `X-API-Key: <API_KEY>` for `/api/*` and `/metrics` |
| `PUBLIC_BIND_REQUIRES_API_KEY`   | `true`                    | security     | Refuse to start if `APP_HOST` is not localhost and `API_KEY` is unset |
| `CORS_ALLOW_ORIGINS`             | `[]`                      | security     | Allowed CORS origins when `DEBUG=false` (JSON list or comma-separated) |
| `RETRIEVAL_MODE`                 | `sparse`                  | retrieval    | `sparse` \| `dense` \| `hybrid`                        |
| `DATA_DIR`                       | `data`                    | storage      | Base data directory (SQLite parent, FAISS paths)       |
| `SQLITE_URL`                     | `sqlite:///./data/app.db` | storage      | SQLite URL                                             |
| `FAQ_CSV`                        | `data/faq.csv`            | ingestion    | FAQ CSV                                                |
| `CSV_HAS_HEADER`                 | `true`                    | ingestion    | CSV has header                                         |
| `INGEST_CHUNK_STRATEGY`          | `chars_v1`                | ingestion    | Chunking strategy identifier (deterministic)           |
| `INGEST_CHUNKER_VERSION`         | `chars_v1`                | ingestion    | Version token included in chunk dedup hashes           |
| `ST_EMBEDDING_MODEL`             | `all-MiniLM-L6-v2`        | dense/hybrid | SentenceTransformers model                             |
| `OPENAI_EMBEDDING_MODEL`         | `text-embedding-3-small`  | OpenAI       | Embeddings model                                       |
| `INDEX_PATH`                     | `data/index.faiss`        | dense/hybrid | FAISS file                                             |
| `ID_MAP_PATH`                    | `data/id_map.json`        | dense/hybrid | FAISS ID map (JSON)                                    |
| `ENABLE_MONITORING`              | `false`                   | monitoring   | Enable metrics middleware and `/metrics` endpoint      |
| `OPENAI_TOP_P`                   | `1.0`                     | OpenAI       | top-p parameter                                        |
| `OPENROUTER_ENABLED`             | `false`                  | OpenRouter   | Enable OpenRouter proxy                                |
| `OPENROUTER_API_KEY`             | —                        | OpenRouter   | API key                                                |
| `OPENROUTER_BASE_URL`            | `https://openrouter.ai/api/v1` | OpenRouter | Base URL                                          |
| `OPENROUTER_MODEL`               | `openai/gpt-4o-mini`     | OpenRouter   | Default model                                         |
| `OPENROUTER_SITE_URL`            | —                        | OpenRouter   | Optional Referer header                                |
| `OPENROUTER_APP_TITLE`           | —                        | OpenRouter   | Optional X-Title header                                |
| `HYBRID_RETRIEVAL_ALPHA`         | `0.5`                     | hybrid       | Weight of the **sparse** component (0=dense, 1=sparse) |
| `OPENAI_API_KEY`                 | —                         | OpenAI       | API key                                                |
| `OPENAI_MODEL`                   | `gpt-4o-mini`             | OpenAI       | Chat model                                             |
| `OPENAI_TEMPERATURE`             | `0.2`                     | OpenAI       | Temperature                                            |
| `OPENAI_MAX_TOKENS`              | `256`                     | OpenAI       | Max tokens                                             |
| `OLLAMA_ENABLED`                 | `false`                   | Ollama       | Enable Ollama                                          |
| `OLLAMA_MODEL`                   | `gemma3:4b`               | Ollama       | Model served by Ollama                                 |
| `OLLAMA_BASE_URL`                | `http://localhost:11434`  | Ollama       | Server URL                                             |
| `OLLAMA_REQUEST_TIMEOUT`         | `180`                     | Ollama       | Timeout (s)                                            |

Example [`.env`](.env.example):

```env
RETRIEVAL_MODE=hybrid
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_TOP_P=1.0
OLLAMA_ENABLED=false
ST_EMBEDDING_MODEL=all-MiniLM-L6-v2
# Optionals
# OPENROUTER_ENABLED=true
# ...
```

### Upgrade notes (SQLite ID integrity)

Dense/hybrid modes rely on document IDs being stable across stores (SQLite + FAISS). SQLite can reuse
integer primary keys after deletes unless `AUTOINCREMENT` is used. On startup, the app will
best-effort migrate legacy `documents` tables to `AUTOINCREMENT` when the schema matches the expected
columns (`id`, `content`). If you have a customized schema, the app will refuse to auto-migrate.

### Upgrade notes (Document identity and metadata)

To support idempotent ingestion and future upserts, the app also ensures the `documents` table contains
stable identity fields and metadata. On startup (SQLite only), it will best-effort add/backfill the
following columns if missing:

* `external_id` (nullable, unique when set): stable document identity for upserts
* `source_id` (nullable): traceability (e.g., filename/url)
* `metadata` (JSON text): structured metadata (best-effort default `{}`)
* `content_sha256`: content hash used by dedup/update policies
* `chunk_dedup_sha256`: optional chunk-level dedup hash (unique when set)
* `created_at`, `updated_at`: timestamps (best-effort backfilled for legacy rows)

---

## Ingestion and indexing flow

* **Sparse**: stores directly in SQLite (no embeddings required).
* **Dense / Hybrid**:

  1. Save chunks in SQLite
  2. Generate embeddings with OpenAI (if `OPENAI_API_KEY`) or SentenceTransformers (`ST_EMBEDDING_MODEL`)
  3. Upsert into FAISS (`INDEX_PATH`, `ID_MAP_PATH`)

Chunking parameters (in settings):

* `INGEST_CHUNK_CHARS` (default 1200)
* `INGEST_CHUNK_OVERLAP` (default 200)
* `INGEST_CHUNKER_VERSION` (default `chars_v1`): changes the dedup key used by `/api/docs` to force re-chunk/re-embed.

Available scripts:

```bash
# Ingest from CSV and build FAISS if applicable
rag-bootstrap


# Ingest .txt/.md/.csv from file(s) or directory(ies)
rag-ingest ./my_notes ./docs/handbook.md ./data/faq.csv


# Explicitly build the index from the CSV (populate SQL and FAISS if applicable)
rag-build-index


# Rebuild FAISS from the current SQLite documents (idempotent; dense/hybrid only)
rag-rebuild-index


# Delete documents by ID from SQLite (and FAISS in dense/hybrid mode)
rag-delete-docs 1 2 3


# Upsert documents by external_id (idempotent)
rag-upsert-docs --external-id doc-1 --content "hello"


# Summarized system and files status
rag-status
```

> Retrieval mode is selected via `RETRIEVAL_MODE` (there is no `--mode` flag).

Optional: better file type detection (best-effort) using `python-magic`:

```bash
uv sync --frozen --extra magic
# or: pip install rag-prototype[magic]
```

### Ingestion pipeline

The ingestion process is orchestrated by `IngestionPipeline`:

1. Load items from a `LoaderPort` (e.g., `CSVLoader`) returning `LoadedItem(text, metadata)`.
2. Preprocess (`preprocess_text`) and chunk (`default_chunker`) with overlap.
3. Format chunks (metadata header) and batch-ingest via `ETLService.ingest()`.

This pipeline is used by `scripts/bootstrap.py`.

---

## LangChain loaders integration (optional)

You can ingest data from any LangChain document loader via the `LangChainLoader` adapter, which implements the project's `LoaderPort`.

Installation:

```bash
uv sync --frozen --extra loaders
# or when installing from PyPI:
# pip install rag-prototype[loaders]
```

Quick usage example:

```python
from langchain_community.document_loaders import WebBaseLoader
from local_rag_backend.core.services.etl import ETLService
from local_rag_backend.core.services.ingestion import IngestionPipeline
from local_rag_backend.infrastructure.ingestion.loaders import LangChainLoader

# 1) Create/obtain your ETLService as usual (doc store, vector store, embedder)
etl = ETLService(doc_repo, vector_repo, embedder)

# 2) Wrap any LangChain loader
lc_loader = WebBaseLoader(["https://example.com"])  # or DirectoryLoader, SitemapLoader, etc.
loader = LangChainLoader(lc_loader, drop_empty=True, metadata_filter={"lang": "en"})

# 3) Run the pipeline
pipeline = IngestionPipeline(loader=loader, etl_service=etl)
count = pipeline.run()
print(f"Ingested {count} chunks")
```

Notes:

- `drop_empty=True` skips whitespace-only documents.
- `metadata_filter={...}` yields only items whose metadata includes the given key/value pairs.
- The adapter expects each LangChain `Document` to have `page_content` and `metadata` fields. It gracefully falls back to dict-like objects or stringification when needed.

---

## Run with Docker Compose (with Ollama)

Prerequisites: Docker Desktop/Engine.

```bash
# Build and start backend + Ollama
docker compose up -d --build

# (Optional) Pull a model into Ollama once the service is up
docker exec -it ollama ollama pull gemma3:4b

# Verify services
curl http://localhost:8000/api/health
curl http://localhost:8000/api/health/ollama
```

Notes:
- Backend listens on `8000`, Ollama on `11434`.
- Configure providers via `.env` or environment variables (see `.env.example`).
- In `docker-compose.yml`, `OLLAMA_ENABLED=true` and `OLLAMA_BASE_URL=http://ollama:11434` are set.
- `docker-compose.yml` defaults to `RETRIEVAL_MODE=sparse` for a lightweight image; install/build with the `dense` extra for dense/hybrid.

---

## API

* `GET /` → Serves packaged `index.html` or the source tree `src/local_rag_backend/frontend/index.html`.
* `GET /api/health` and `GET /api/ready`
* `GET /api/health/ollama`
* `GET /api/config` and `GET /api/templates`
* `POST /api/ask`

  * Body: `{ "question": "str", "k": int (1..10, default 3) }`
  * Response: `{ "answer": "str", "sources": [ { "document": {"id": int, "content": "str"}, "score": float(0..1) }, ... ] }`
* `GET /api/history?limit=1..100&offset>=0`

  * Response: list of `{ id, question, answer, created_at, source_ids[] }`
* FastAPI docs: `GET /docs` and `GET /openapi.json`
* `POST /api/docs` (ingest texts) and `GET /api/docs` (list docs)
* `POST /api/docs/upsert` (idempotent upsert by `external_id`)
* `POST /api/docs/delete` (delete docs by ID; keeps SQL + FAISS consistent when applicable)
* `POST /api/index/rebuild` (idempotent rebuild of FAISS from SQLite; dense/hybrid only)
* `POST /api/openrouter/generate` (enabled if OpenRouter configured)

Notes:

* Retrieval “scores” are normalized to [0,1] in the adapters.
* The service persists each Q/A with the IDs of the retrieved sources.
* In dense/hybrid mode, **FAISS is derived state**; use `/api/docs/delete` (or `rag-delete-docs`) instead of deleting rows manually.
* In dense/hybrid mode, `/api/ready` is intentionally strict and returns `503` when it detects missing/corrupt index files or drift between SQLite documents and the vector index (hinting how to rebuild).

Example:

```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "k": 3}'
```

---

## Tests

```bash
pytest
# Coverage (already enabled by default via pyproject.toml)
# pytest --cov-report=xml:coverage.xml
# pytest --cov-report=html:htmlcov

# If your environment has a read-only/non-writable home directory, prefer:
make test
```

> Test suite includes unit, integration, and E2E (FastAPI TestClient). Some integration tests require `faiss` and/or `sentence_transformers`; if they are not installed, those tests are skipped automatically. The suite enforces `--cov-fail-under=85` via `pyproject.toml`.

## Documentation site (optional)

If you install docs extras (`uv sync --extra docs`), you can run:

```bash
mkdocs serve
```

---

## Extension and integration points

* **LLM**: implement `GeneratorPort` (see `infrastructure/llms/*`) and wire it in `app/factory.py`.
* **Retriever**: implement `RetrieverPort` and wire it in `factory.get_retriever()`.
* **Vector store**: implement `VectorRepoPort` (e.g., an alternative to FAISS).
* **Document store**: implement `DocumentRepoPort` to use a DB other than SQLite.
* **Loader**: implement `LoaderPort` for new sources (PDFs, web, etc.).

---

## Runtime considerations

* **Singleton per process**: `RagService` is initialized as a singleton in `factory`. With `uvicorn --workers N`, each process loads its own instance (and its FAISS). Align deployment and warm-up as needed.
* **Metrics**: if `ENABLE_MONITORING=true` and `prometheus-client` is installed, `/metrics` provides Prometheus format.
* **Dense/Hybrid**: must use the same embedding model for indexing and querying (`ST_EMBEDDING_MODEL`).

---

## Current limitations

* Synchronous LLM clients (requests/OpenAI SDK); migration to async is straightforward but not included.
* Minimal UI without front-end tests.
* Minimal API-key auth is available (`API_KEY`), but there is no user/role authZ or rate limiting.
* FAISS index type `IndexFlatL2` (simple). For large volumes, consider IVF/HNSW or other backends.

---

## License

MIT. See [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by [Intrinsical AI](https://intrinsical.ai)**

[📝 Report Issues](https://github.com/Intrinsical-AI/rag-prototype/issues) • [💬 Discussions](https://github.com/Intrinsical-AI/rag-prototype/discussions)

</div>

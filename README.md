# Intrinsical RAG Prototype

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-125%20passing-brightgreen.svg)](https://github.com/Intrinsical-AI/rag-prototype/actions)
[![Coverage](https://img.shields.io/badge/coverage-85%25-green.svg)](https://github.com/Intrinsical-AI/rag-prototype)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/r/intrinsical/rag-prototype)
[![PyPI](https://img.shields.io/pypi/v/intrinsical-rag-prototype.svg)](https://pypi.org/project/intrinsical-rag-prototype/)
[![Downloads](https://img.shields.io/pypi/dm/intrinsical-rag-prototype.svg)](https://pypi.org/project/intrinsical-rag-prototype/)

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
├── frontend/                  # Simple UI (index.html) for dev
├── src/local_rag_backend/
│   ├── app/                   # FastAPI (main, routers, DI, factory)
│   ├── core/                  # domain, ports and services (ETL, RAG)
│   ├── infrastructure/        # adapters: llms, retrievers, storage, loaders
│   ├── scripts/               # bootstrap and build_index
│   └── frontend/              # packaged index.html to serve at /
└── tests/                     # unit + integration + e2e
```

---

## Requirements

* Python 3.11+
* Operating system: Linux / macOS / Windows
* For dense/hybrid mode: `faiss` and `sentence_transformers` (installed as extras or manually)

---

## Installation and startup (from source)

```bash
git clone https://github.com/Intrinsical-AI/rag-prototype.git
cd rag-prototype


python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate


# Install the package (add extras if you want faiss/sentence_transformers)
pip install -e .


# (Optional) Install development dependencies
# pip install -e ".[dev]"
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
> `python -m local_rag_backend.scripts.bootstrap` and
> `uvicorn local_rag_backend.app.main:app --reload`.

---

## Configuration

> Options are in `local_rag_backend/settings.py` (Pydantic Settings). They can be overridden with environment variables or a `.env` file (case-insensitive).

| Variable                         | Default                   | Scope        | Description                                            |
| -------------------------------- | ------------------------- | ------------ | ------------------------------------------------------ |
| `APP_HOST`                       | `0.0.0.0`                 | server       | Service host                                           |
| `APP_PORT`                       | `8000`                    | server       | Service port                                           |
| `DEBUG`                          | `false`                   | server       | Reload/detailed logging                                |
| `RETRIEVAL_MODE`                 | `hybrid`                  | retrieval    | `sparse` \| `dense` \| `hybrid`                        |
| `SQLITE_URL`                     | `sqlite:///./data/app.db` | storage      | SQLite URL                                             |
| `FAQ_CSV`                        | `data/faq.csv`            | ingestion    | FAQ CSV                                                |
| `CSV_HAS_HEADER`                 | `true`                    | ingestion    | CSV has header                                         |
| `ST_EMBEDDING_MODEL`             | `all-MiniLM-L6-v2`        | dense/hybrid | SentenceTransformers model                             |
| `INDEX_PATH`                     | `data/index.faiss`        | dense/hybrid | FAISS file                                             |
| `ID_MAP_PATH`                    | `data/id_map.pkl`         | dense/hybrid | FAISS ID map                                           |
| `ENABLE_MONITORING`              | `false`                   | monitoring   | Enable metrics middleware and `/metrics` endpoint      |
| `OPENAI_TOP_P`                   | `1.0`                    | OpenAI       | top-p parameter                                        |
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
| `OLLAMA_ENABLED`                 | `false`                   | Ollama       | Enable Ollama                                          |
| `OLLAMA_MODEL`                   | `gemma3:1b`               | Ollama       | Model served by Ollama                                 |
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

---

## Ingestion and indexing flow

* **Sparse**: stores directly in SQLite (no embeddings required).
* **Dense / Hybrid**:

  1. Save chunks in SQLite
  2. Generate embeddings with SentenceTransformers (`ST_EMBEDDING_MODEL`)
  3. Upsert into FAISS (`INDEX_PATH`, `ID_MAP_PATH`)

Chunking parameters (in settings):

* `INGEST_CHUNK_CHARS` (default 1200)
* `INGEST_CHUNK_OVERLAP` (default 200)

Available scripts:

```bash
# Ingest from CSV and build FAISS if applicable
rag-bootstrap


# Explicitly build the index from the CSV (populate SQL and FAISS if applicable)
rag-build-index


# Summarized system and files status
rag-status
```

> Retrieval mode is selected via `RETRIEVAL_MODE` (there is no `--mode` flag).

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
pip install -e ".[loaders]"
# or when installing from PyPI:
# pip install intrinsical-rag-prototype[loaders]
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
docker exec -it ollama ollama pull gemma:2b

# Verify services
curl http://localhost:8000/api/health
curl http://localhost:8000/api/health/ollama
```

Notes:
- Backend listens on `8000`, Ollama on `11434`.
- Configure providers via `.env` or environment variables (see `.env.example`).
- In `docker-compose.yml`, `OLLAMA_ENABLED=true` and `OLLAMA_BASE_URL=http://ollama:11434` are set.

---

## API

* `GET /` → Serves packaged `index.html` or the repo’s `frontend/index.html`.
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
* `POST /api/openrouter/generate` (enabled if OpenRouter configured)

Notes:

* Retrieval “scores” are normalized to [0,1] in the adapters.
* The service persists each Q/A with the IDs of the retrieved sources.

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
# Coverage
pytest --cov=src --cov-report=term-missing
```

> Tests suite includes unit, integration, and E2E (FastAPI TestClient). Some integration tests require `faiss` and/or `sentence_transformers`; if they are not installed, those tests are skipped automatically. Current status: **125 tests, 86.45% coverage**.

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
* No authentication/rate limiting or exported metrics (logging and status CLI are included).
* FAISS index type `IndexFlatL2` (simple). For large volumes, consider IVF/HNSW or other backends.

---

## License

MIT. See [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by [Intrinsical AI](https://intrinsical.ai)**

[📝 Report Issues](https://github.com/Intrinsical-AI/rag-prototype/issues) • [💬 Discussions](https://github.com/Intrinsical-AI/rag-prototype/discussions)

</div>

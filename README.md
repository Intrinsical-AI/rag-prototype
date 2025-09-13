# Intrinsical RAG Prototype

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-78%20passing-brightgreen.svg)](https://github.com/Intrinsical-AI/rag-prototype/actions)
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
* **Tests**

  * Unit, integration, and E2E with `pytest`.

---

## Project structure

```


.
├── data/                      # CSV, SQLite DB, FAISS files
├── frontend/                  # Simple UI (index.html) for dev
├── src/local\_rag\_backend/
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
* For web ingestion: optional extra `loaders` (installs `trafilatura`)

---

## Installation and startup (from source)

```bash
git clone https://github.com/Intrinsical-AI/rag-prototype.git
cd rag-prototype


python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate


# Install the package (add extras if you want faiss/sentence_transformers)
pip install -e .

# (Optional) Web ingestion support (trafilatura)
# pip install -e ".[loaders]"


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
# Ollama healthcheck: http://localhost:8000/api/health/
# Docs: http://localhost:8000/docs
```

> If you prefer to invoke scripts directly:
> `python -m local_rag_backend.scripts.bootstrap` and
> `uvicorn local_rag_backend.app.main:app --reload`.

---

## Configuration

Options are in `local_rag_backend/settings.py` (Pydantic Settings). They can be overridden with environment variables or a `.env` file (case-insensitive).

| Variable                         | Default                   | Scope        | Description                                            |
| -------------------------------- | ------------------------- | ------------ | ------------------------------------------------------ |
| `APP_HOST`                       | `0.0.0.0`                 | server       | Service host                                           |
| `APP_PORT`                       | `8000`                    | server       | Service port                                           |
| `DEBUG`                          | `false`                   | server       | Reload/detailed logging                                |
| `RETRIEVAL_MODE`                 | `sparse`                  | retrieval    | `sparse` \| `dense` \| `hybrid`                        |
| `SQLITE_URL`                     | `sqlite:///./data/app.db` | storage      | SQLite URL                                             |
| `FAQ_CSV`                        | `data/faq.csv`            | ingestion    | FAQ CSV                                                |
| `CSV_HAS_HEADER`                 | `true`                    | ingestion    | CSV has header                                         |
| `ST_EMBEDDING_MODEL`             | `all-MiniLM-L6-v2`        | dense/hybrid | SentenceTransformers model                             |
| `INDEX_PATH`                     | `data/index.faiss`        | dense/hybrid | FAISS file                                             |
| `ID_MAP_PATH`                    | `data/id_map.pkl`         | dense/hybrid | FAISS ID map                                           |
| `ENABLE_FAISS_CONSISTENCY_CHECK` | `true`                    | dense/hybrid | FAISS↔SQL check on startup                             |
| `HYBRID_RETRIEVAL_ALPHA`         | `0.5`                     | hybrid       | Weight of the **sparse** component (0=dense, 1=sparse) |
| `OPENAI_API_KEY`                 | —                         | OpenAI       | API key                                                |
| `OPENAI_MODEL`                   | `gpt-3.5-turbo`           | OpenAI       | Chat model                                             |
| `OPENAI_TEMPERATURE`             | `0.2`                     | OpenAI       | Temperature                                            |
| `OLLAMA_ENABLED`                 | `false`                   | Ollama       | Enable Ollama                                          |
| `OLLAMA_MODEL`                   | `gemma3:1b`               | Ollama       | Model served by Ollama                                 |
| `OLLAMA_BASE_URL`                | `http://localhost:11434`  | Ollama       | Server URL                                             |
| `OLLAMA_REQUEST_TIMEOUT`         | `90`                      | Ollama       | Timeout (s)                                            |

Example `.env`:

```env
RETRIEVAL_MODE=hybrid
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-3.5-turbo
OLLAMA_ENABLED=false
ST_EMBEDDING_MODEL=all-MiniLM-L6-v2
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
# Ingest from CSV and, if applicable, build FAISS
rag-bootstrap


# Explicitly build the index from the CSV (populate SQL and optionally FAISS)
rag-build-index


# Summarized system and files status
rag-status


# Ingest from filesystem (patterns, optional dedup & incremental)
rag-ingest-files --root ./docs --pattern "**/*.md" --pattern "**/*.txt"


# Ingest from web (requires extra [loaders])
rag-ingest-web --url https://example.com --url https://example.org


# Health check (wires RAG service like the API would)
rag-health
```

> Retrieval mode is selected via `RETRIEVAL_MODE` (there is no `--mode` flag).

---

## API

* `GET /` → Serves packaged `index.html` or the repo’s `frontend/index.html`.
* `POST /api/ask`

  * Body: `{ "question": "str", "k": int (1..10, default 3) }`
  * Response: `{ "answer": "str", "sources": [ { "document": {"id": int, "content": "str"}, "score": float(0..1) }, ... ] }`
* `GET /api/history?limit=1..100&offset>=0`

  * Response: list of `{ id, question, answer, created_at, source_ids[] }`
* `GET /docs` and `GET /openapi.json`

Notes:

* Retrieval “scores” are normalized to \[0,1] in the adapters.
* The service persists each Q/A with the IDs of the retrieved sources.

Example:

```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "k": 3}'
```

---

## Filesystem and Web ingestion via CLI

These commands ingest and index content end-to-end using the same preprocessing/chunking as the CSV bootstrap.

#### Filesystem examples

```bash
# Basic: all markdown and text files under a directory
rag-ingest-files --root ./docs

# Custom patterns (repeatable)
rag-ingest-files --root . --pattern "**/*.md" --pattern "**/*.txt" --pattern "**/*.rst"

# Disable in-run deduplication
rag-ingest-files --root ./notes --no-dedup

# Incremental ingestion (default): skips unchanged files using a state file
rag-ingest-files --root ./knowledge --state-path data/.ingest_state.json

# Incremental based on content hash (robust, slightly slower)
rag-ingest-files --root ./knowledge --incremental-strategy hash
```

Notes:

* Deduplication is per-run and in-memory (same text within this run is skipped). It does not persist across runs.
* Incremental state is stored as JSON (default `data/.ingest_state.json`) with per-file `mtime` and `sha1` (if hash strategy). Delete the file to force full re-ingestion.

#### Web ingestion examples (requires extra [loaders])

```bash
# Install optional dependency
pip install -e ".[loaders]"

# Single or multiple URLs
rag-ingest-web --url https://en.wikipedia.org/wiki/Retrieval-augmented_generation \
               --url https://en.wikipedia.org/wiki/Vector_space_model

# From a file (one URL per line)
rag-ingest-web --from-file urls.txt

# Increase concurrency
rag-ingest-web --from-file urls.txt --workers 8

# Disable in-run deduplication
rag-ingest-web --from-file urls.txt --no-dedup
```

#### Quickstart: Wikipedia/blog

```bash
cat > urls.txt << 'EOF'
https://en.wikipedia.org/wiki/Retrieval-augmented_generation
https://en.wikipedia.org/wiki/Vector_space_model
https://en.wikipedia.org/wiki/FAISS
EOF

pip install -e ".[loaders]"
rag-ingest-web --from-file urls.txt --workers 6

# Start the API and ask a question
rag-server --reload
curl -s -X POST "http://localhost:8000/api/ask" -H 'Content-Type: application/json' \
  -d '{"question": "What is FAISS and how is it used in RAG?", "k": 3}' | jq
```

---

## Troubleshooting (ingestion)

* Trafilatura missing (web ingestion): install the optional extra
  - `pip install -e ".[loaders]"`
  - Error message: `trafilatura is not installed. Install the 'loaders' extra: pip install intrinsical-rag-prototype[loaders]`
* Windows: SentenceTransformers may require Microsoft Visual C++ Redistributable
  - Ensure it is installed if you see DLL-related errors when using dense/hybrid retrieval
* FAISS vs CPU/GPU
  - The project uses `faiss-cpu` by default. Do not install `faiss-gpu` simultaneously. See `[project.optional-dependencies].performance` in `pyproject.toml` for GPU notes
* Network environments
  - `rag-ingest-web` performs HTTP requests; configure proxies via standard environment variables if needed
  - Use `--timeout` to adjust request timeouts
* Reset incremental state
  - Delete `data/.ingest_state.json` (or your custom `--state-path`) to force a full re-ingestion
  - Switch strategy with `--incremental-strategy mtime|hash`

---

## Tests

```bash
pytest
# Coverage
pytest --cov=src --cov-report=term-missing
```

The suite includes unit, integration, and E2E (FastAPI TestClient). Some integration tests require `faiss` and/or `sentence_transformers`; if they are not installed, those tests are skipped automatically.

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
* **FAISS↔SQL consistency**: can optionally be validated on startup (`ENABLE_FAISS_CONSISTENCY_CHECK`). For large collections, you may disable it.
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

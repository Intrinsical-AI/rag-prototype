# Architecture Guide: Hexagonal (Ports & Adapters)

The **Intrinsical RAG Prototype** uses a **Hexagonal architecture** (a.k.a. Ports & Adapters) to maximize modularity, testability, and maintainability. Business logic lives in the core; external tech (LLMs, vector stores, databases) are plugged in via adapters.

## Core Principles

* **Dependency Inversion**: the core depends on *ports* (interfaces), never on concrete implementations.
* **Stable Core**: domain entities and services are tech-agnostic.
* **Adapters at the Edge**: infrastructure code implements the ports.
* **Composition Root**: `app/container.py` composes adapters/use-cases; `app/factory.py` manages runtime app context and compatibility entrypoints, using shared selection helpers from `app/composition.py`.
* **Testability**: adapters can be swapped for fakes/mocks; ports are `Protocol`s.

---

## Project Structure

```
src/local_rag_backend/
├── cli_commands/               # CLI commands partitioned by domain (docs/index/eval/server)
├── core/                       # Domain + application services (technology-agnostic)
│   ├── domain/                 # Entities (Document, etc.)
│   ├── ports/                  # Ports (Protocols) for core dependencies
│   └── services/               # ETL, IngestionPipeline, RagService
├── infrastructure/             # Adapters (technology-specific)
│   ├── embeddings/             # ST/OpenAI embedders
│   ├── llms/                   # OpenAI / Ollama generators
│   ├── persistence/            # sql/, vector/, shared/
│   ├── retrieval/              # BM25 (sparse), dense vector, Hybrid
│   └── ingestion/              # CSV loader, etc.
├── app/                        # Application + HTTP transport layer
│   ├── main.py                 # FastAPI app + lifespan
│   ├── api_router.py           # Root API router composition (include_router only)
│   ├── routers/                # HTTP handlers by bounded context
│   ├── http/                   # HTTP-only concerns (exception handlers, transport boundary)
│   ├── application/            # Use-case orchestration (transport-agnostic)
│   ├── dependencies.py         # DI bridge to app context/container
│   ├── app_context.py          # Runtime context (settings + AppContainer)
│   ├── container.py            # App composition container used by routers/CLI
│   ├── schemas/                # Pydantic request/response schemas by bounded context
│   ├── diagnostics.py          # Readiness/status diagnostics used by API/CLI
│   ├── composition.py          # Shared adapter selection policy (embedder/retriever/generator)
│   ├── factory.py              # App-context lifecycle + compatibility entrypoints
│   └── services/               # Use-case orchestration + app-layer contracts/results
└── scripts/                    # CLI helpers (bootstrap, build_index)
```

Evaluation layering:
- `core/services/evaluation.py` is technology-agnostic (dataset parsing + metric computation).
- `app/services/evaluation.py` owns ephemeral SQL/retriever wiring for `rag-eval`.

HTTP docs/index layering:
- Routers are thin adapters in `app/routers/*` and call use-case orchestration in
  `app/application/*` and `app/services/*`.
- Shared mutation execution (`run_api_mutation` / `run_cli_mutation`) lives in
  `app/application/mutations.py`.
- App-layer dependency contracts for docs/index mutations live in `app/services/ports.py`
  (`DocsMutationPorts`, `IndexMutationPorts`), with shared wiring in
  `app/services/mutation_ports.py`.
- App-layer docs mutation outcomes live in `app/services/results.py`
  (`UpsertDocsSummary`, `DeleteDocsSummary`, etc.).

Type taxonomy (enforced by module naming):
- `app/schemas/*`: HTTP transport contracts (Pydantic only).
- `app/services/results.py` and `core/services/types.py`: internal DTOs/results (transport-agnostic).
- `core/domain/entities.py`: domain entities and invariants.
- `infrastructure/persistence/*/models.py`: ORM persistence models.

Error layering:
- Infra adapters raise typed runtime errors from `core/errors.py` (no FastAPI dependency).
- Runtime error mapping lives in `app/error_mapping.py`.
- HTTP registration/rendering lives in `app/http/exception_handlers.py`.

CLI layering:
- `cli.py` is the entrypoint module (group + command registration).
- Shared command runtime helpers live in `cli_commands/runtime.py`.
- Domain commands live in `cli_commands/docs.py`, `cli_commands/index.py`,
  `cli_commands/eval.py`, and `cli_commands/server.py`.

---

## Dependency Flow (High-Level)

```mermaid
graph TD
  A[FastAPI Router] --> B[RagService]
  B --> C[RetrieverPort]
  B --> D[GeneratorPort]
  B --> E[QAHistoryPort]

  subgraph Core
    B
    C
    D
    E
  end

  subgraph Infrastructure (Adapters)
    C1[SparseBM25Retriever]
    C2[DenseVectorRetriever]
    C3[HybridRetriever]
    D1[OpenAIGenerator]
    D2[OllamaGenerator]
    E1[HistorySqlStorage]
    S1[SqlDocumentStorage]
    V1[VectorStorage]
  end

  A -->|DI via factory| C1 & C2 & C3 & D1 & D2 & E1 & S1 & V1
```

The composition root (`app/container.py` + `app/factory.py`) chooses specific adapters
(BM25/FAISS/Hybrid; OpenAI/Ollama) using `settings.py`, with policy centralized in
`app/composition.py` and reused by API/CLI/scripts.

---

## Key Ports (Core Interfaces)

Source of truth: `src/local_rag_backend/core/ports/__init__.py`.

```python
# src/local_rag_backend/core/ports/__init__.py
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from local_rag_backend.core.domain.entities import Document, Embedding, LoadedItem

@runtime_checkable
class EmbedderPort(Protocol):
    dim: int
    def embed(self, texts: Sequence[str]) -> Sequence[Embedding]: ...

@runtime_checkable
class GeneratorPort(Protocol):
    def generate(self, question: str, contexts: Sequence[str]) -> str: ...

@runtime_checkable
class RetrieverPort(Protocol):
    def retrieve(self, query: str, k: int = 5) -> tuple[Sequence[Document], Sequence[float]]: ...

@runtime_checkable
class DocumentRepoPort(Protocol):
    def store_documents(self, contents: Sequence[str]) -> Sequence[int]: ...
    def delete_documents(self, ids: Sequence[int]) -> None: ...
    def get(self, ids: Sequence[int]) -> Sequence[Document]: ...
    def get_all_documents(self) -> Sequence[Document]: ...

@runtime_checkable
class VectorRepoPort(Protocol):
    def upsert(self, ids: Sequence[int], vectors: Sequence[Embedding]) -> None: ...
    def delete(self, ids: Sequence[int]) -> int: ...
    def rebuild(self, ids: Sequence[int], vectors: Sequence[Embedding]) -> None: ...
    def similar(self, vector: Embedding, k: int) -> Sequence[tuple[int, float]]: ...

@runtime_checkable
class QAHistoryPort(Protocol):
    def save(self, q: str, a: str, source_ids: Sequence[int]) -> None: ...

@runtime_checkable
class LoaderPort(Protocol):
    def load(self) -> Iterable[LoadedItem]: ...
```

---

## Representative Adapters

**Retrievers**

* `SparseBM25Retriever` (BM25 over preprocessed text; SQL for doc lookup)
* `DenseVectorRetriever` (SentenceTransformers/OpenAI embeddings + FAISS; SQL for doc lookup)
* `HybridRetriever` (linear blend of dense + sparse, configurable `alpha`)

**LLMs**

* `OpenAIGenerator` (chat completions)
* `OllamaGenerator` (HTTP to local Ollama server)

**Persistence**

* `SqlDocumentStorage` (documents via SQLAlchemy/SQLite)
* `VectorStorage` (vector index + ID map)
* `HistorySqlStorage` (Q\&A history)

**App transport**

* Pydantic HTTP schemas live in `src/local_rag_backend/app/schemas/` (split by bounded context: `rag`, `docs`, `index`, `meta`).

**Ingestion**

* `CSVLoader` → `IngestionPipeline` → `ETLService` (store docs, embed, upsert vectors)

All of these implement the ports above and can be swapped at composition time.

---

## Composition Root

`app/container.py` wires the system from configuration (factory/providers + runtime cache),
while `app/factory.py` owns app-context lifecycle and compatibility accessors.

* Chooses **retriever** by `settings.retrieval_mode` (`sparse`, `dense`, `hybrid`)
* Chooses **generator**: Ollama (if `OLLAMA_ENABLED`) or OpenAI (if `OPENAI_API_KEY`)
* Instantiates `RagService(retriever, generator, history_storage)`
* Provides a process-local singleton via `get_rag_service()`
* Cross-process cache invalidation uses DB-backed `system_state.version` (key: `rag_service`),
  bumped by `reset_rag_service()`

---

## Request Flow (End-to-End)

```mermaid
sequenceDiagram
  participant U as Client
  participant API as FastAPI /api/ask
  participant S as RagService
  participant R as RetrieverPort
  participant G as GeneratorPort
  participant H as QAHistoryPort

  U->>API: POST /api/ask {"question": "...", "k": 3}
  API->>S: ask(question, top_k=k)
  S->>R: retrieve(query, k)
  R-->>S: (docs, scores)
  alt no docs
    S-->>API: {"answer": "No hay documentos indexados para responder a tu pregunta.", "sources": []}
  else docs
    S->>G: generate(question, [doc.content...])
    G-->>S: answer
    S->>H: save(question, answer, source_ids=[...])
    S-->>API: {"answer": answer, "sources": [{document, score}, ...]}
  end
```

`/api/history` reads persisted Q\&A with pagination.

Async/sync boundary:
- FastAPI handlers call sync core/infra paths via `app/blocking.py`.
- Blocking work is partitioned by task type (`default`, `mutation`, `network`, `eval`) with
  dedicated worker pools and queue limits to reduce event-loop starvation risk.

---

## Multi-Store Consistency (SQLite + FAISS)

In dense/hybrid retrieval, the system has **two stores**:

* **SQLite** (`documents` table) is the source of truth for document text.
* **FAISS** (`INDEX_PATH` + `ID_MAP_PATH`) is derived state: it maps `document_id -> embedding vector`.

### Document Identity Contract

The `documents` table is designed to support idempotent ingestion and upserts (already available via API and CLI):

* `id`: internal integer primary key (stable due to SQLite `AUTOINCREMENT`)
* `external_id`: optional stable identifier for a source document (unique when set)
* `source_id`: optional provenance identifier (e.g., file path, URL)
* `metadata`: JSON metadata captured at extraction time (stored as JSON text in SQLite)
* `content_sha256`: hash of the stored `content` (dedup/update decisions)
* `created_at`, `updated_at`: timestamps

These fields allow you to track and update documents without relying on brittle “row order” or
manual deletion. Current user-facing upsert flows are:

* `POST /api/docs/upsert`
* `rag-upsert-docs`

The invariants that matter:

* Document IDs must be stable (SQLite uses `AUTOINCREMENT` to avoid ID reuse after deletes).
* Writes must keep SQL and FAISS consistent, or fall back to a safe recovery path.

Maintenance logic lives in `src/local_rag_backend/core/services/maintenance.py`:

* `delete_documents_multi_store(...)`:
  * preflights index mutability (`vec_repo.delete([])`) before SQL delete when rebuild fallback would require a not-yet-resolved embedder,
  * aborts before SQL mutation if preflight fails and no embedder is available for safe rebuild,
  * otherwise deletes from SQLite, attempts vector deletion, and falls back to full rebuild when configured.
* `rebuild_index_from_db(...)`: idempotent rebuild of FAISS from the current SQLite docs.

Because the application caches a process-local singleton `RagService`, API/CLI maintenance operations call `reset_rag_service()` after mutating the DB and/or index so subsequent queries see the updated state.

---

## HTTP API Surface

The API root router lives in `src/local_rag_backend/app/api_router.py` (mounted under `/api`) and
includes bounded routers from `src/local_rag_backend/app/routers/`.
The project exposes:

* `POST /api/docs` and `GET /api/docs` (ingest/list documents)
* `POST /api/docs/upsert` (idempotent upsert by `external_id`)
* `POST /api/docs/delete` (delete docs by ID; keeps SQL + FAISS consistent when applicable)
* `POST /api/docs/delete_by_external_id` (delete by `external_id` + tombstones)
* `POST /api/index/rebuild` (idempotent rebuild of FAISS from SQLite; dense/hybrid only)
* `POST /api/ask_eval` (ephemeral per-request RAG configuration)
* `POST /api/openrouter/generate` (OpenRouter proxy when configured)
* `GET /api/config` and `GET /api/templates`
* `GET /api/health`, `GET /api/ready`, `GET /api/health/ollama`

For current request/response shapes, prefer the OpenAPI schema at `GET /openapi.json` (or `GET /docs` in dev).

---

## Extending the System

**Add a new retriever (e.g., Elasticsearch):**

1. Implement `RetrieverPort`.
2. Resolve documents (by ID) via your `DocumentRepoPort` implementation.
3. Expose a setting (e.g., `RETRIEVAL_MODE=elasticsearch`) and branch in `factory.py`.

**Add a new LLM (e.g., Anthropic):**

1. Implement `GeneratorPort`.
2. Add settings (API key, model, etc.).
3. Select in `factory.get_generator()` based on settings.

**Swap embeddings backend:**

* Implement `EmbedderPort` (or reuse `OpenAIEmbedder` / `SentenceTransformerEmbedder`).
* Ensure FAISS index dimensionality matches `embedder.dim`.
* Rebuild the index after changing the embedding model.

---

## Testing Strategy

* **Unit tests**: mock the ports to isolate core services (`RagService`, `ETLService`, `IngestionPipeline`).
* **Integration tests**: real SQLite (temp), optional FAISS, real BM25; adapters tested together.
* **E2E tests**: FastAPI `TestClient` hitting `/api/ask` and `/`.

The codebase already includes fixtures (e.g., in-memory SQLite with `StaticPool`), adapter fakes, and coverage for edge cases (dim mismatches, missing docs, error propagation).

---

## Trade-offs & Considerations

* More files/indirection than a simple script, but greatly improved swapability and testability.
* FAISS `IndexFlatL2` is chosen for simplicity; for larger corpora, consider IVF/HNSW and external vector DBs.
* The RAG prompt templates live in settings; adapt them to your safety/grounding needs.

---

## Minimal Code Examples

**Port usage in a service (core):**

```python
# src/local_rag_backend/core/services/rag.py
class RagService:
    def __init__(self, retriever, generator, history):
        self.retriever = retriever
        self.generator = generator
        self.history = history

    def ask(self, question: str, top_k: int = 3):
        docs, scores = self.retriever.retrieve(question, top_k)
        if not docs:
            answer = "No hay documentos indexados para responder a tu pregunta."
            self.history.save(question, answer, [])
            return {"answer": answer, "docs": [], "scores": []}
        answer = self.generator.generate(question, [d.content for d in docs])
        self.history.save(question, answer, [d.id for d in docs])
        return {"answer": answer, "docs": docs, "scores": scores}
```

**Adapter implementing a port (sparse example):**

```python
# src/local_rag_backend/infrastructure/retrieval/sparse_bm25.py
class SparseBM25Retriever(RetrieverPort):
    def __init__(self, documents, doc_ids, doc_repo):
        # tokenize+fit BM25; keep doc_repo to resolve IDs -> Document
        ...

    def retrieve(self, query: str, k: int = 5):
        # BM25 scores -> normalize -> map to Document via repo -> return (docs, scores)
        ...
```

**Container wiring:** see `src/local_rag_backend/app/container.py` and
`src/local_rag_backend/app/factory.py`.

# Architecture Guide: Hexagonal + Durable Multi-Store Writes

`rag-prototype` uses a hexagonal architecture (Ports & Adapters) with explicit layer boundaries and a transport-neutral composition root.

## Core principles

- Core domain logic depends on ports, never on infrastructure implementations.
- HTTP/CLI are transport adapters; business orchestration lives in `core/use_cases/`.
- Persistence and model providers are swappable through ports and wiring factories.
- Multi-store writes (SQL + vector index) are centralized in one coordinator (`DURABLE_SAGA`).
- FastAPI is an optional dependency (`[server]` extra). The core, infrastructure, and composition layers work without it.

> Golden Rule: Si borras la carpeta `http/` (FastAPI), el sistema RAG (ingesta, mutación, query) debe seguir funcionando al 100% usando solo Python puro invocando `core/services`.

---

## Current structure

```text
src/local_rag_backend/
├── core/
│   ├── domain/                 # entities, value objects, storage profiles
│   ├── ports/                  # Embedder/Generator/Retriever/Repo ports + contracts
│   ├── services/               # domain-oriented services (rag_runtime, ingestion, etc.)
│   └── use_cases/              # transport-agnostic use cases/orchestration
│       ├── docs_mutation.py    # MutationCoordinator (canonical write path)
│       ├── docs_ingest.py      # ingest texts use case
│       ├── docs_import.py      # import JSON (ChatGPT/Gemini) use case
│       ├── docs_query.py       # query/list docs use case
│       ├── rag_query.py        # ask_eval + history read path
│       ├── mutations.py        # shared API/CLI mutation execution wrapper
│       ├── errors.py           # typed app errors + map_runtime_error
│       └── results.py          # use-case output DTOs
├── infrastructure/
│   ├── persistence/            # sql + vector + shared
│   ├── retrieval/              # sparse/dense/hybrid adapters
│   ├── llms/                   # OpenAI/Ollama adapters
│   ├── embeddings/             # OpenAI/ST adapters
│   ├── concurrency/            # blocking task executor (stdlib-only)
│   └── observability/          # telemetry, metrics, diagnostics
├── composition/                # DI container, factory, wiring (transport-neutral)
│   ├── container.py            # AppContainer composition root
│   ├── factory.py              # app context lifecycle
│   ├── adapters.py             # infrastructure adapter builders
│   └── wiring/                 # default wiring builders for mutation ports
├── http/                       # FastAPI transport adapter (optional [server] extra)
│   ├── routers/                # HTTP route handlers
│   ├── schemas/                # Pydantic HTTP models
│   ├── main.py                 # ASGI entry point
│   └── ...                     # middleware, security, dependencies
└── cli_commands/               # CLI transport adapters
```

---

## Layer dependency rules

```
core/domain/    ← imported by all, imports nothing from local_rag_backend
core/ports/     ← imports core/domain/ only
core/services/  ← imports core/domain/, core/ports/
core/use_cases/ ← imports core/domain/, core/ports/, core/services/
infrastructure/ ← imports core/ only
composition/    ← imports core/, infrastructure/ (http/ only under TYPE_CHECKING)
http/           ← imports core/, infrastructure/, composition/
cli_commands/   ← imports core/, composition/ (NOT http/)
```

These boundaries are enforced by architecture tests under `tests/unit/http/test_architecture_*`.

---

## Layer boundaries

- `core/{domain,ports,services}` must not import `infrastructure/`, `http/`, or `composition/`.
- `core/use_cases/` must not import `http/` or `fastapi`/`starlette`.
- `http/routers/*` must not import `infrastructure/*` directly.
- `composition/` only imports `http/` under `TYPE_CHECKING`.
- `cli_commands/` imports `core/` and `composition/`, never `http/`.
- Cross-layer runtime error mapping is centralized in `core/use_cases/errors.py::map_runtime_error` + `http/exception_handlers.py`.

### Transport isolation contract

**Golden rule**: deleting the entire `http/` directory must not break the RAG system.
Ingestion, mutation, and query must remain fully functional by importing and calling
`core/services` directly from plain Python — no FastAPI, no Starlette, no HTTP.

Practical test — the following must work in a vanilla Python script:

```python
from local_rag_backend.core.services.rag_runtime import RagService
from local_rag_backend.core.use_cases.docs_mutation import MutationCoordinator
from local_rag_backend.core.domain.profiles import StorageProfileRegistry
```

**Implication for use-case authors**: functions in `core/use_cases/` must accept
transport-neutral inputs — `LoaderPort`, `Sequence[str]`, `bytes`, or `io.BytesIO` —
never `fastapi.UploadFile` or Pydantic HTTP schemas. The FastAPI router converts the
HTTP request into those neutral types before calling the use case.

---

## Multi-store write model

### Why

In dense/hybrid modes, writes affect:

- SQL documents (entity store),
- vector index (operational index).

Physical atomicity across both stores is not assumed. The system guarantees `DURABLE_SAGA` semantics through journaling, compensation, and recovery.

### Canonical write path

All document mutations must go through:

- `MutationCoordinator.execute(...)`

Used by:

- `POST /api/docs/mutate`
- `POST /api/docs` ingestion flow (internally builds mutation intents)
- `POST /api/docs/import` flow (internally builds mutation intents)
- `rag-mutate-docs`
- `rag-ingest`

### Write capabilities

`StorageProfile` declares capabilities:

- `ATOMIC`
- `DURABLE_SAGA`
- `READ_ONLY`

Writes are rejected if the active profile does not satisfy `DURABLE_SAGA` for write-enabled modes.

### Mutation states

Journal records move through:

- `PREPARED`
- `SQL_COMMITTED`
- `VECTOR_COMMITTED`
- `COMMITTED`
- `COMPENSATING`
- `ROLLED_BACK`
- `FAILED_NEEDS_RECOVERY`

### Normal write flow

1. Acquire multi-store write lock (`WRITE_LOCK_TIMEOUT_S` / `WRITE_LOCK_POLL_S`).
2. Journal `PREPARED`.
3. Capture SQL `before_image`.
4. Apply SQL mutation and commit.
5. Journal `SQL_COMMITTED`.
6. Compute vector delta from changed content only.
7. Apply vector delta atomically (`VectorRepoPort.apply_delta_atomic`).
8. Journal `VECTOR_COMMITTED`.
9. Journal `COMMITTED` and cleanup.

### Failure handling

If vector apply fails after SQL commit:

1. Journal `COMPENSATING`.
2. Roll back SQL via `before_image`.
3. Mark `ROLLED_BACK` if success, otherwise `FAILED_NEEDS_RECOVERY`.

Startup and background recovery loops replay incomplete journal records until convergence.

---

## API surface (v1.0)

Kept:

- `GET /api/docs`
- `POST /api/docs`
- `POST /api/docs/import`
- `POST /api/docs/mutate`
- `POST /api/index/rebuild`
- `POST /api/ask`
- `POST /api/ask_eval`
- `GET /api/history`
- `GET /api/health`
- `GET /api/ready`

Removed:

- `POST /api/docs/upsert`
- `POST /api/docs/delete`
- `POST /api/docs/delete_by_external_id`

---

## CLI surface (v1.0)

Kept:

- `rag-ingest`
- `rag-mutate-docs`
- `rag-rebuild-index`
- `rag-bootstrap`
- `rag-status`
- `rag-eval`
- `rag-server`

Removed:

- `rag-upsert-docs`
- `rag-delete-docs`
- `rag-delete-external-ids`

---

## Swappability contract

The architecture assumes backends can be swapped (e.g., SQL-only vector plugin, Postgres+Milvus, Elastic+vector). Current hard requirements for write-enabled dense/hybrid profiles:

- document repo supports mutation primitives used by `MutationCoordinator`,
- vector adapter implements:

```python
def apply_delta_atomic(
    *,
    delete_ids: Sequence[DocId],
    upserts: Sequence[tuple[DocId, Embedding]],
) -> None: ...
```

If a vector adapter does not implement `apply_delta_atomic`, writes fail closed.

---

## Operational guidance

- Treat rebuild as explicit repair only:
  - `POST /api/index/rebuild`
  - `rag-rebuild-index`
- Monitor readiness:
  - index drift/corruption checks,
  - mutation journal incomplete-record warnings.
- Do not mutate SQL/vector stores independently in normal operation.

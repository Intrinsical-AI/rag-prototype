# Architecture Guide: Hexagonal + Durable Multi-Store Writes

`rag-prototype` uses a hexagonal architecture (Ports & Adapters) with an explicit app layer for orchestration and transport boundaries.

## Core principles

- Core domain logic depends on ports, never on infrastructure implementations.
- HTTP/CLI are transport adapters; business orchestration lives in `app/application`.
- Persistence and model providers are swappable through ports and wiring factories.
- Multi-store writes (SQL + vector index) are centralized in one coordinator (`DURABLE_SAGA`).

---

## Current structure

```text
src/local_rag_backend/
├── app/
│   ├── application/            # transport-agnostic use cases/orchestration
│   │   ├── docs_mutation.py    # MutationCoordinator (canonical write path)
│   │   ├── docs_*_use_case.py  # ingest/import/query docs use cases
│   │   ├── rag_query_use_case.py
│   │   ├── mutations.py        # shared API/CLI mutation execution wrapper
│   │   └── storage_profiles.py # capability gates (ATOMIC/DURABLE_SAGA/READ_ONLY)
│   ├── contracts/              # app-layer contracts (ports/results)
│   ├── wiring/                 # default wiring builders for mutation ports
│   ├── routers/                # FastAPI transport adapters
│   ├── schemas/                # HTTP Pydantic models
│   ├── container.py            # composition root for app runtime
│   └── factory.py              # app context lifecycle and compatibility shims
├── core/
│   ├── domain/                 # entities/value objects
│   ├── ports/                  # Embedder/Generator/Retriever/Repo ports
│   └── services/               # domain-oriented services (rag_runtime, ingestion, etc.)
├── infrastructure/
│   ├── persistence/            # sql + vector + shared
│   ├── retrieval/              # sparse/dense/hybrid adapters
│   ├── llms/                   # OpenAI/Ollama adapters
│   └── embeddings/             # OpenAI/ST adapters
└── cli_commands/               # CLI transport adapters
```

`app/services` does not exist anymore by design. Its old responsibilities were split into:

- `app/contracts/*` for contracts/DTOs.
- `app/wiring/*` for adapter wiring.
- `app/application/*` for use-case orchestration.

---

## Layer boundaries

- `app/routers/*` must not import `infrastructure/*` directly.
- `app/application/*` must not import HTTP transport (`fastapi`, routers, schemas).
- `core/*` must not import `app/*` or concrete infrastructure.
- Cross-layer runtime error mapping is centralized in `app/error_mapping.py` + `app/http/exception_handlers.py`.

These boundaries are enforced by architecture tests under `tests/unit/app/test_architecture_*`.

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

# rag-prototype — Architecture Spec

> **Owners:** Intrinsical-AI maintainers (core + platform)
> **Scope:** core architecture, boundaries, write/read flows, contracts, quality gates / **Out of scope:** product roadmap/features, UI design, legacy backward compatibility

---

## 0) TL;DR (90 seconds)

- **What:** `rag-prototype` is a local-first RAG system (library + CLI + optional FastAPI transport) built as a modular monolith with hexagonal boundaries. The system supports sparse/dense/hybrid retrieval, SQL persistence, and vector index backends (FAISS/numpy) with consistency controls.
- **Why:** The immediate priority is architectural stabilization and decoupling (not feature expansion). We will accept breaking changes to eliminate structural debt now and freeze a clean first deliverable.
- **How:**
  - Domain + ports in `core/`, adapters in `infrastructure/`, composition in `composition/`, transports in `http/` and `cli_commands/`.
  - Canonical write path via `MutationCoordinator` (`DURABLE_SAGA`) for SQL + vector consistency.
  - App/runtime wiring centralized in `AppContainer` + `composition/*`.
  - Architecture guard tests enforce import boundaries in CI.
  - Runtime safety with write lock, mutation journal, startup/background recovery.
- **Non-negotiables:**
  - All write-enabled document mutations go through `core/use_cases/docs_mutation.py::MutationCoordinator`.
  - No direct writes to SQL/vector stores outside canonical mutation/index use cases.
  - `core/{domain,ports,services}` must not depend on `infrastructure/`, `http/`, `composition/`.
  - Target state: `core/use_cases` depends on ports/contracts only (no concrete infra imports). [TODO: pending full implementation; current violations exist in `docs_query`, `rag_query`, `health`, `evaluation`, `openrouter`, `mutations`]
  - Breaking changes are allowed during this deliverable; no migration compatibility layer is required.
  - CI gates (`pre-commit`, `ruff`, `mypy`, `pytest`, architecture tests) are mandatory.

---

## 1) Goals, Non-goals, Constraints

### 1.1 Goals (prioritized)
1. Close Deliverable D1: architecture stabilization + maximum practical decoupling.
2. Enforce architectural guardrails in code (not only docs).
3. Reduce blast radius in high-complexity modules (`docs_mutation`, `alchemy_engine`, composition wiring).
4. Freeze a clear, maintainable baseline for subsequent product work.

### 1.2 Non-goals
- Adding new user-facing features.
- Preserving old internal APIs or legacy module paths.
- Multi-service split (keep modular monolith).
- Backward compatibility with external clients from previous iterations.

### 1.3 Constraints (hard)
- Legal / regulatory: no special regulated-domain requirement declared for D1.
- Budget / latency / throughput: local-first single-node operation, optional Docker; avoid infra-heavy dependencies by default.
- Team: small maintainer set; changes must be reviewable in small increments.
- Tech (languages, runtime, hosting): Python 3.11/3.12, `uv`, FastAPI optional extra (`server`), SQLite + vector index on disk.
- Product constraint: breaking changes explicitly allowed for this deliverable.

### 1.4 Quality attributes

| Attribute    | Target | Measured by |
| ------------ | -----: | ----------- |
| Latency p95 (`POST /api/ask`, sparse, local sample dataset) | <= 2000ms | `rag_query_duration_seconds` metric + e2e smoke |
| Availability (single instance) | >= 99.0% in controlled environment | `/api/health` + `/api/ready` checks |
| Consistency | SQL atomic + SQL/vector `DURABLE_SAGA` | mutation journal recovery tests + integration tests |
| Cost | single-node local runtime (no mandatory external SaaS except optional LLM provider) | Docker/local runtime footprint |

### 1.5 Definition of Done (Deliverable D1)
- Guardrails:
  - Architecture tests enforce target dependencies including `core/use_cases` decoupling objective. [TODO: extend current architecture tests to enforce this explicitly]
  - No forbidden imports remain in target modules. [TODO: pending cleanup in listed `core/use_cases` modules]
- Decoupling:
  - Critical use cases (`docs_query`, `rag_query`, `health`, `evaluation`, `openrouter`) consume ports/adapters abstractions, not concrete infra modules. [TODO: pending refactor]
- Stability:
  - CI baseline green (`pre-commit`, `ruff`, `mypy`, `pytest`). [TODO: keep as release gate after decoupling changes]
  - Mutation consistency and recovery smoke tests pass. [TODO: keep as mandatory smoke suite during refactor]
- Documentation:
  - This spec reflects actual code boundaries and entrypoints.
  - Known tradeoffs and accepted debt are explicit.

---

## 2) Domain model

### 2.1 Glossary (ubiquitous language)
- **Document:** Core textual unit retrievable by RAG (`core/domain/entities.py::Document`).
- **DocId:** Stable domain identifier (`core/domain/types.py::DocId`).
- **External ID:** Stable source identity used for idempotent upsert/delete semantics.
- **LoadedItem:** Ingestion unit from a loader with lineage metadata.
- **MutationIntent:** Requested write operation (upserts/deletes) for canonical mutation flow.
- **MutationRecord:** Durable journal state for multi-store writes.
- **StorageProfile:** Capability profile (`ATOMIC`, `DURABLE_SAGA`, `READ_ONLY`) for allowed operations.
- **Retriever mode:** `sparse`, `dense`, `hybrid` retrieval strategy.

### 2.2 Bounded contexts

| Context | Responsibility | Owns data? | External deps | Public APIs |
| ------- | -------------- | ---------: | ------------- | ----------- |
| Retrieval Query | Retrieve docs + generate answer | No | LLM adapters, retrievers | `/api/ask`, `/api/ask_eval`, `RagService.ask` |
| Document Mutation | Canonical write orchestration SQL + vector | Yes | SQL repo, vector repo, embedder | `/api/docs`, `/api/docs/import`, `/api/docs/mutate`, `rag-mutate-docs`, `rag-ingest` |
| Index Maintenance | Rebuild/repair vector index | Yes | embedder + vector adapter | `/api/index/rebuild`, `rag-rebuild-index` |
| Health/Diagnostics | Readiness/consistency diagnostics | No | SQL engine, manifest/index files | `/api/health`, `/api/ready`, `rag-status` |
| Transport (HTTP/CLI) | Input/output mapping + auth + error translation | No | FastAPI/Click | REST + CLI commands |
| Composition Runtime | Dependency wiring + runtime cache invalidation | No | settings + adapters | DI factory/container |

### 2.3 Entities & value objects

#### Entity: `Document`
- **Identity:** `DocId` (`doc:<uuid7>`)
- **Lifecycle:** created -> updated (content/metadata/source) -> deleted (optional tombstone by `external_id`)
- **Invariants:** see section 4
- **Context:** Document Mutation / Retrieval Query
- **Storage:** `documents` table

#### Entity: `MutationRecord`
- **Identity:** `op_id`
- **Lifecycle:** `PREPARED -> SQL_COMMITTED -> VECTOR_COMMITTED -> COMMITTED` with compensation paths (`COMPENSATING`, `ROLLED_BACK`, `FAILED_NEEDS_RECOVERY`)
- **Invariants:** see section 4
- **Context:** Document Mutation
- **Storage:** file-backed mutation journal (`data/.mutation_journal`)

#### Entity: `QaHistory`
- **Identity:** auto-increment integer
- **Lifecycle:** append-only question/answer records
- **Invariants:** question and answer non-null
- **Context:** Retrieval Query
- **Storage:** `qa_history` table

#### Value Object: `DocId`
- **Equality:** by string value
- **Validation:** must be non-empty when consumed by repos/use cases

#### Value Object: `ItemLineage`
- **Equality:** structural equality of source and transform metadata
- **Validation:** source URI + loader metadata must be serializable

#### Value Object: `StorageProfile`
- **Equality:** by `profile_id` + capabilities set
- **Validation:** profile must exist in registry (`StorageProfileRegistry`)

---

## 3) Data model

### 3.1 Storage overview
- Primary DB: SQLite via SQLAlchemy.
- Cache: in-process runtime cache (`RagService` cache versioned via `system_state`).
- Search / vector index: FAISS or numpy index backend on disk.
- Files / blobs: local filesystem (`data/`, index artifacts, mutation journal).

### 3.2 Schemas
- SQL models: `src/local_rag_backend/infrastructure/persistence/sql/models.py`
- SQL compatibility/bootstrap: `src/local_rag_backend/infrastructure/persistence/sql/base.py`
- Vector manifest contract: `src/local_rag_backend/infrastructure/persistence/vector/manifest.py`

### 3.3 DTOs / Contracts
- HTTP DTOs: `src/local_rag_backend/http/schemas/`
- Use-case outputs: `src/local_rag_backend/core/use_cases/results.py`
- Mutation/index app contracts: `src/local_rag_backend/core/ports/contracts.py`
- Versioning: semver at package level; D1 allows breaking internal contracts to reach clean boundaries.

#### DTO: `AskRequest` / `AskResponse`
- **Purpose:** HTTP request/response for RAG question answering
- **Fields:**
  - request: `question:str`, `k:int`
  - response: `answer:str`, `sources:[document+score]`
- **Backward compat:** not guaranteed during D1 stabilization
- **Example:**

```json
{
  "question": "What is durable saga?",
  "k": 3
}
```

### 3.4 Mapping rules (DTO <-> Domain <-> Persistence)
- DTO -> Domain: in `http/routers/*`, with request validation by Pydantic schemas.
- Domain -> Persistence: via ports and adapters (`DocumentRepoPort`, `VectorRepoPort`, `QAHistoryPort`).
- Forbidden shortcuts:
  - HTTP routers importing concrete infra adapters directly.
  - Domain/services embedding framework-specific request/response models.
  - Non-canonical SQL/vector writes that bypass mutation coordinator.

---

## 4) Invariants & validation

### 4.1 Global invariants
- Multi-store write operations are serialized under shared write lock.
- Canonical write flow is journaled and recoverable (`DURABLE_SAGA`).
- Dense/hybrid mutation cannot silently proceed when embeddings backend is unavailable.
- Retrieval mode must be one of `sparse|dense|hybrid`.
- Transport isolation rule: removing `http/` must not break core mutation/query capabilities.

### 4.2 Per-aggregate invariants

| Aggregate | Invariant | Enforced where | Test coverage |
| --------- | --------- | -------------- | ------------- |
| Document | `external_id` uniqueness, tombstone semantics for deleted external IDs | SQL model + mutation use case + repo methods | unit + integration |
| MutationRecord | valid state machine transitions + idempotent `op_id` behavior | `MutationCoordinator` + journal adapter | unit + integration |
| Vector index mapping | no drift between SQL docs and id map/manifest in steady state | diagnostics + readiness checks + maintenance flows | unit + integration + e2e |

### 4.3 Failure semantics
- Validation errors: mapped to `AppError` hierarchy (`400/401/404/409/413/422/5xx`).
- Idempotency: `op_id` in mutation intent guarantees replay-safe behavior.
- Retry safety: incomplete mutation records are recoverable at startup/background intervals.
- Consistency model per operation:
  - SQL-only operations: atomic per DB transaction.
  - SQL + vector mutations: `DURABLE_SAGA` with compensation/recovery.

---

## 5) Architecture style & layers

### 5.1 Style
- Hexagonal architecture inside a modular monolith.
- Rationale:
  - keep domain/use-case logic independent of frameworks and concrete providers;
  - allow optional transport (`FastAPI` extra) and swappable adapters.
- Tradeoffs:
  - more wiring/ports overhead;
  - risk of boundary drift if guardrails are weak.

### 5.2 Layers

| Layer | Responsibilities | Must NOT contain |
| ----- | ---------------- | ---------------- |
| Domain (`core/domain`) | entities, value objects, storage profile semantics | HTTP/DB/LLM framework code |
| Application (`core/use_cases`, `core/services`) | use cases, orchestration, business flow | transport framework dependencies |
| Adapters (`infrastructure`) | DB, vector index, embeddings, LLM clients, observability adapters | domain policy decisions |
| Composition (`composition`) | dependency assembly, runtime lifecycle/wiring | business rules |
| Transport (`http`, `cli_commands`) | request parsing, auth, response mapping, command UX | domain invariants |

### 5.3 Dependency rules

```text
Domain       -> (nothing project-internal outside domain/types)
Application  -> Domain + Ports
Adapters     -> Application contracts + Domain
Composition  -> Application + Adapters
UI/API/CLI   -> Application + Composition
```

Rule: no upward imports. Enforced with architecture tests and CI.
[TODO: add explicit guardrail for `core/use_cases` -> `infrastructure` imports, not only `core/{domain,ports,services}`.]

### 5.4 Target ports to decouple `core/use_cases`
[TODO: define and implement these ports in `core/ports/contracts.py` (or dedicated application contracts module), then migrate use-cases incrementally.]

| Use case module | Current coupling | Target port/abstraction |
| --------------- | ---------------- | ----------------------- |
| `docs_query.py` | direct SQL model import | `DocsReadPort` (`list_docs_page`) |
| `rag_query.py` | direct SQL storage/crud imports | `HistoryReadPort` + `RagRuntimeFactoryPort` |
| `health.py` | direct diagnostics/sql/vector imports | `HealthDiagnosticsPort` |
| `evaluation.py` | direct SQL/retriever construction | `EvalStoragePort` + `EvalRetrieverFactoryPort` |
| `openrouter.py` | direct OpenAI client factory import | `ChatCompletionPort` (OpenAI-compatible adapter) |
| `mutations.py` | direct blocking import | `BlockingExecutorPort` |

---

## 6) Components & interactions

### 6.1 Component diagram
- Source of truth diagram path (to maintain in D1): `docs/diagrams/components.mmd`
- Runtime key components:
  - `RagService` (query orchestration)
  - `MutationCoordinator` (write orchestration)
  - `AppContainer` (composition root)
  - SQL repository adapters
  - Vector repository adapters
  - LLM/Embedding adapters

### 6.2 Main flows

#### Flow: `Ask query`
- Trigger: `POST /api/ask` or library call to `RagService.ask`
- Steps: resolve service -> retrieve docs -> generate answer -> persist history -> return answer + sources
- Side effects: history row insert
- Failure modes: provider timeout/error, retrieval errors, malformed provider response

#### Flow: `Canonical mutation`
- Trigger: `/api/docs/mutate`, `/api/docs`, `/api/docs/import`, `rag-mutate-docs`, `rag-ingest`
- Steps: normalize intent -> acquire lock -> journal -> SQL mutation -> vector delta -> commit -> cleanup
- Side effects: SQL writes, vector index updates, mutation journal entries
- Failure modes: vector failure after SQL commit, rollback failure, lock timeout

#### Flow: `Index rebuild/repair`
- Trigger: `/api/index/rebuild` or `rag-rebuild-index`
- Steps: purge artifacts -> embed all SQL docs -> rebuild vector index -> emit diagnostics
- Side effects: full index rewrite
- Failure modes: embedder unavailable, manifest/drift mismatch, index persistence failure

---

## 7) Interfaces: APIs, events, commands

### 7.1 Public API
- Protocol: REST + CLI + Python library usage.
- Auth: optional `X-API-Key`, plus safe-bind enforcement for non-local requests.
- Rate limiting: no explicit built-in limiter currently (must be handled by deployment edge if needed).
- HTTP endpoints (current):
  - `/api/ask`, `/api/ask_eval`, `/api/history`
  - `/api/docs`, `/api/docs/import`, `/api/docs/mutate`
  - `/api/index/rebuild`
  - `/api/health`, `/api/ready`, `/api/health/ollama`
  - `/api/openrouter/generate`

### 7.2 Events & messaging
- Broker: none (no async message bus in current architecture).
- Topics: none.
- Delivery: synchronous request/response only.
- Consumer idempotency: achieved in mutation path via `op_id` + journal, not via broker semantics.

---

## 8) Security, privacy, compliance

- Threat model: accidental public exposure of costly/mutating endpoints.
- Secrets management: environment variables (`.env` for local dev), never hardcoded.
- PII handling: avoid raw question logging; logs use hashed question fingerprint (`q_sha256`, `q_len`).
- Log redaction: structured logs avoid full payloads by default.
- Supply-chain:
  - pinned lockfile (`uv.lock`)
  - pinned CI actions
  - `gitleaks`, `bandit`, `safety`, `pre-commit` gates.

---

## 9) Performance, scalability, capacity

- Workload assumptions:
  - single-node service
  - local SQLite and local disk index
  - moderate concurrent requests
- Bottlenecks:
  - embedding/generation network latency
  - index rebuild operations
  - shared write lock under heavy mutation load
- Benchmarks:
  - functional: `pytest` integration/e2e suites
  - runtime signals: `rag_query_duration_seconds`, blocking queue metrics
- Scaling strategy:
  - vertical first (single instance)
  - optional multi-worker ASGI; process cache invalidation via `system_state` versioning
  - no distributed storage orchestration in D1 scope

---

## 10) Observability

- Logs: JSON-style structured event logs (`telemetry.log_event`).
- Metrics (golden signals and domain):
  - `rag_queries_total`, `rag_query_duration_seconds`
  - `rag_ingest_requests_total`, `rag_ingest_docs_total`
  - `rag_blocking_pending_tasks`, `rag_blocking_saturation_ratio`, queue wait/run histograms
  - optional HTTP metrics via middleware when monitoring is enabled
- Tracing: no distributed tracing backend integrated yet.
- Alerting thresholds (initial recommendation):
  - readiness failures > 0 in rolling window
  - blocking saturation ratio > 0.8 sustained
  - mutation journal incomplete records > 0 sustained

---

## 11) Testing strategy

| Type | Scope | Tools | Required gates |
| ---- | ----- | ----- | -------------- |
| Unit | domain/services/use cases/adapters | `pytest` | pass |
| Integration | SQL/vector/index/recovery interactions | `pytest` | pass |
| E2E | API behavior and retrieval smoke | `pytest` | pass |
| Architecture | import boundaries/layer rules | `pytest` + AST checks | pass |
| Type/Lint/Sec | static quality | `mypy`, `ruff`, `bandit`, `safety`, `pre-commit` | pass |

Mandatory CI gate: total coverage >= 85% (`pytest` config in `pyproject.toml`).

---

## 12) Deployment & environments

- Environments: local dev, CI, containerized runtime.
- Config strategy: 12-factor with `pydantic-settings` + env vars.
- Migrations/compatibility:
  - SQLite compatibility ensured at startup/CLI bootstrap.
  - D1 allows breaking schema/contracts if required for decoupling.
- Rollback plan:
  - code rollback via git release tags
  - data/index repair via rebuild and mutation recovery
  - fail-closed behavior preferred when consistency is at risk

---

## 13) Repo layout & conventions

```text
src/local_rag_backend/
  core/
    domain/
    ports/
    services/
    use_cases/
  infrastructure/
  composition/
  http/
  cli_commands/
tests/
docs/
  architecture.md
```

- Naming: explicit `*_port`, `*_factory`, `*_use_case` semantics.
- Error handling: typed app errors in `core/use_cases/errors.py`, mapped at transport boundary.
- Code style: `ruff` + `mypy` strict profile; avoid broad exceptions unless justified for recovery/rollback guards.

---

## 14) Architecture decisions (ADRs)

ADRs must include alternatives, tradeoffs, consequences.

Planned ADR set for D1 (to create under `docs/adr/`):
1. ADR-001: Canonical mutation path and `DURABLE_SAGA` as mandatory write model.
2. ADR-002: `core/use_cases` decoupling policy and required ports.
3. ADR-003: Single composition root strategy and wiring ownership.
4. ADR-004: Settings/config access policy (avoid hidden global coupling in app/core paths).
5. ADR-005: Architecture test suite as release gate.

---

## 15) Notes for new developers

- How to run locally:
  - `uv venv .venv && source .venv/bin/activate`
  - `uv sync --frozen --extra server --group test --group lint --no-default-groups`
  - `rag-bootstrap`
  - `rag-server`
- How to run tests:
  - full: `UV_CACHE_DIR=.uv_cache DEBUG=false uv run pytest -q`
  - architecture smoke: `UV_CACHE_DIR=.uv_cache DEBUG=false uv run pytest -q -o addopts='' tests/unit/http/test_architecture_*.py`
- Where to add a new feature:
  - domain rules in `core/domain` or `core/services`
  - orchestration in `core/use_cases`
  - I/O implementation in `infrastructure`
  - expose via `http/routers` or `cli_commands`
- Common pitfalls:
  - importing concrete infrastructure from use cases
  - bypassing `MutationCoordinator` for write paths
  - relying on global settings in deep layers without explicit dependency injection
  - running partial test subsets without overriding global coverage `addopts`

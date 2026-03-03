# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [Unreleased]
- importlinter as arch deps/imports watcher. Safety check added to CI + precommit.

## [1.3.0] - 2026-03-03

### Breaking

- Removed legacy SQL shim module `src/local_rag_backend/infrastructure/persistence/sql/alchemy_engine.py`
  and its remaining compatibility surface.

### Added

- Golden integration coverage for invariants F1-F5 in
  `tests/integration/test_invariants_f1_f5_golden.py`:
  - F1/F2 durable mutate + ingest + list contracts.
  - F3 ask + ask_eval + history contracts.
  - F4/F5 rebuild + health/readiness contracts.

### Changed

- Composition wiring is now centralized through `AppContainer.runtime_wiring_defaults()` with
  `AppContainer.from_settings()` as the single runtime construction path.
- Factory/container wiring overrides were unified to reduce duplication across API/CLI/runtime paths.
- Bootstrap ingestion (`rag-bootstrap`) now goes through canonical durable mutation flow
  (`MutationCoordinator`) instead of the legacy ETL-style path.
- Bootstrap/integration naming and architecture docs were updated to reflect the canonical mutation
  topology and single composition root.

### Fixed

- `composition.factory` now resolves runtime wiring overrides with safe defaults and keeps
  monkeypatchable public symbols for test/runtime overrides.
- Lock/offload assertions were hardened in mutation tests to check behavior instead of fragile
  internal callable names.

## [1.2.0] - 2026-02-26

### BREAKING

- **Screaming Architecture refactor**: `app/` package eliminated entirely.
  - `app/application/` → `core/use_cases/` (transport-agnostic use cases).
  - `app/contracts/ports.py` → `core/ports/contracts.py`.
  - `app/contracts/results.py` → `core/use_cases/results.py`.
  - `app/errors.py` → `core/use_cases/errors.py`.
  - `app/{composition,container,factory,app_context}.py` → `composition/` (transport-neutral DI root).
  - `app/wiring/` → `composition/wiring/`.
  - `app/{diagnostics,metrics_backend,observability,telemetry}.py` → `infrastructure/observability/`.
  - `app/blocking.py` → `infrastructure/concurrency/blocking.py`.
  - `app/application/storage_profiles.py` → `core/domain/profiles.py`.
  - All remaining HTTP-specific files → `http/` (routers, schemas, middleware, security, main).
- **FastAPI is now an optional dependency**: moved `fastapi`, `uvicorn`, `python-multipart` to
  `[project.optional-dependencies.server]`. Install with `pip install rag-prototype[server]`
  or `uv sync --extra server`.
- ASGI entry point changed: `local_rag_backend.app.main:app` → `local_rag_backend.http.main:app`.
- `all` extra now uses PEP 621 self-referencing syntax to include `server` transitively.
- `error_mapping.py` moved from `http/` into `core/use_cases/errors.py::map_runtime_error`.
- `AppContainer` public method signatures use `AskEvalConfigLike` Protocol (from
  `core/use_cases/rag_query`) instead of the concrete `AskEvalConfig` Pydantic schema.

### Added

- Architecture guard tests enforce layer dependency rules:
  - `core/{domain,ports,services}` must not import `infrastructure/`, `http/`, or `composition/`.
  - `core/use_cases/` must not import `http/` or `fastapi`/`starlette`.
  - No `local_rag_backend.app.*` references remain in source.
- `AskEvalConfigLike` Protocol with 9 properties for transport-neutral RAG evaluation config.
- `Makefile` sync targets include `--extra server` for dev/test/lint environments.

### Changed

- Dockerfile: ASGI entry point updated to `local_rag_backend.http.main:app`; `--extra server`
  added to dependency sync stages.

## [1.1.0] - 2026-02-25

### Security
- Bumped starlette 0.37.2 → 0.50.0 (via fastapi ≥ 0.124): resolves three DoS CVEs
  (CVE-2024-47874, CVE-2025-54121, CVE-2025-62727).
- Bumped anyio 4.3 → 4.12: closes PVE-2024-71199.

### Added
- `VectorRepoPort.ntotal` read-only property for non-destructive preflight health checks
  (used in maintenance preflight before rebuild/delete operations).
- `WriteLockPort` Protocol in `core/ports/contracts.py`; `DocsMutationPorts.write_lock`
  is now explicitly typed against it.
- gitleaks v8.24.2 pre-commit hook for secret and credential detection.
- `check-json` and `debug-statements` pre-commit hooks.

### Changed
- fastapi pin raised to `>=0.124,<0.125` (minimum required to unlock starlette ≥ 0.50).
- python-multipart declared as an explicit dependency (was transitive in fastapi ≤ 0.111).
- `DocsMutationPorts`: four dead fields removed (`precompute_vectors_fn`, `sync_dense_fn`,
  `delete_docs_fn`, `delete_external_ids_fn`).
- `DocsRepositoryPort`: extended with `get()`, `delete_documents()`, `delete_by_external_ids()`.
- `UpsertDocBuilderPort`: return type narrowed from `Any` to `object`.
- `MultiStoreDeleteResult` and `MultiStoreExternalIdDeleteResult` are now frozen dataclasses;
  `maintenance.py` functions no longer return bare tuples.
- `core/services/types.py` replaces `core/services/schemas.py` for data-only DTOs.
- `VectorStorage.__init__` accepts an explicit `settings_obj` parameter instead of
  relying on the global settings singleton.

### Fixed
- `except Exception` narrowed to specific types (`UnicodeDecodeError`, `json.JSONDecodeError`)
  in `factory.py` and `id_map_json.py`; broad catches in `alchemy_engine.py` and
  `vector/index.py` are documented as intentionally wide (rollback/recovery guards).
- `docs_ingest.py` no longer imports `SqlDocumentStorage` directly; uses the
  `ports.build_upsert_doc` factory instead.
- `sample_data_ingestion.py`: `print()` replaced by `logger.info()`; magic constant
  `128` replaced by `settings_obj.ingest_batch_size`.
- All production `assert x is not None` replaced by explicit `RuntimeError`.
- Orphaned result DTOs removed: `UpsertDocsSummary`, `DeleteDocsByExternalIdSummary`,
  `DeleteDocsSummary`.
- `runtime.__all__` no longer exposes private `_reset_rag_service_best_effort`.
- `mutation_journal._record_from_dict`: invalid-state entries now log a warning before
  resetting to `PREPARED` instead of silently resetting.
- `except TypeError` removed from `write_lock.py` and `docs_mutation.py`.

### Refactor
- `_resolve_embedder_or_raise()` helper extracted in `maintenance.py`, removing four
  duplicated preflight patterns across two functions.
- `FaissIndex._locked_write()` context manager replaces five repeated lock patterns.
- `Settings.ingest_batch_size` field added; magic number `64` removed from callers.
- `IngestPlan` and `BatchSyncResult` converted from tuple aliases to frozen dataclasses.
- `_to_domain_document()` and `_detect_document_changes()` extracted in `sql_.py`
  (eliminates `getattr` usage; adds explicit change detection).
- `DocsMutationPorts` / `IndexMutationPorts`: `Any` replaced by concrete types
  throughout mutation port contracts.
- `factory.py:get_app_context` dead local variable for mypy narrowing removed.

### Docs
- README: FastAPI badge corrected (0.111+ → 0.124+); `cli_commands/` added to project
  structure tree; `scripts/` directory description corrected.
- USAGE.md: `sessionmaker(bind=engine)` → `sessionmaker(engine)` (SQLAlchemy 2.x API).

## [1.0.0] - 2026-02-24

### Added
- New ingestion pipeline for files/directories with loader discovery for `.txt`, `.md`, and `.csv`, plus optional `python-magic` detection.
- New CLI capabilities: `rag-ingest`, `rag-delete-external-ids`, and `rag-eval` (dataset-based offline regression gate).
- New API mutation capabilities for document lifecycle and maintenance (`/api/docs/upsert`, `/api/docs/delete_by_external_id`, `/api/index/rebuild`).
- Manifest-backed dense index diagnostics with drift checks surfaced in readiness/status flows.
- Optional overlap reranker and expanded observability (structured logs + ingestion/query metrics).
- DB-backed `system_state` versioning to invalidate cached RAG services across processes.
- Task-type blocking pools with explicit pending limits for mutation/network/eval workloads.

### Changed
- Release line normalized for `release/02-2026` at `v1.0.0`.
- API transport split into bounded routers (health/rag/docs/index/openrouter) with composition-focused root wiring.
- Mutation orchestration moved to app-layer services and shared mutation ports reused by API and CLI.
- CLI reorganized by bounded contexts while preserving command surface.
- Runtime composition policy centralized for retriever/embedder/provider resolution across API, CLI, and scripts.
- Default Ollama model consolidated to `lfm2.5-thinking` and ingestion batching formalized via `INGEST_BATCH_SIZE`.

### Fixed
- Multi-store consistency hardening: dense embeddings are precomputed before SQL upserts to avoid SQL/vector drift on provider failures.
- Mutating operations now serialize under a shared cross-process write lock and fail closed on lock acquisition errors.
- FAISS/index consistency hardening: manifest preflight checks, stricter lock behavior, safer persistence and recovery paths.
- Ingestion correctness fixes: external_id prefix collision handling, symlink no-follow behavior, duplicate external_id validation, and robust unreadable-file handling.
- API hardening fixes: malformed OpenRouter responses mapped cleanly (502), stricter sampling parameter validation, and deterministic cache invalidation after mutation attempts.
- SQLite compatibility hardening: identity migration race tolerance and safer compatibility migrations for CLI/scripts.

### Security
- Runtime API-key enforcement for non-local exposure, including forwarded/proxied requests (`X-Forwarded-*` / `Forwarded`) with fail-closed behavior on ambiguity.
- Additional validation guards for generation/evaluation request parameters and non-local request handling.
- CI/security posture hardened (workflow gate tightening, pinned actions, and security-check workflow improvements).

### Performance
- Ingestion batching optimizations to reduce write-lock and upsert churn.
- Sparse retrieval hot-path optimization via in-memory document caching and reduced duplicate SQL loads.
- Reduced ingestion overhead by reusing file-format detection results and precompiling whitespace cleanup regexes.

### Refactor
- Removal of root shims and stricter app/core module boundaries.
- Typed cross-layer provider errors and centralized HTTP error mapping.
- Consolidated dense upsert/delete consistency flow and shared locking/client helpers.
- RAG service invalidation refactored from file token strategy to DB-backed versioning (`system_state`).

### Docs
- Updated architecture and usage guides for bounded routers/services, new CLI/API mutation flows, eval workflow, and observability.
- Documented manifest drift behavior, ingestion dedup/chunking strategy, delete-by-external-id semantics, and contributor verification gates.
- Added/updated operational notes for rebuild/delete maintenance and release stabilization roadmap.

### Migration / Upgrade notes
- Upgrading from `0.1.x`: run app/CLI once per SQLite database so compatibility migrations can add identity and consistency fields.
- Dense/hybrid mode now depends on manifest integrity (`index_manifest.json`); if readiness reports drift/corruption, run index rebuild.
- Delete semantics changed: deleting by `external_id` creates tombstones and blocks future re-ingest/upsert of those identities.
- Ingestion dedup now uses `chunk_dedup_sha256` + `INGEST_CHUNKER_VERSION`; changing chunker version intentionally creates new chunk identities.
- Review production env before rollout: set `API_KEY` for non-local binds and tune ingestion concurrency with `INGEST_BATCH_SIZE`.
- Release tag for this cut: `v1.0.0` on `release/02-2026`.

## [0.1.2] - 2026-02-16

### Added
- Optional API key auth via `API_KEY` (clients must send `X-API-Key`) to protect `/api/*` and `/metrics`.
- Production-safe CORS allowlist via `CORS_ALLOW_ORIGINS` when `DEBUG=false`.
- Metrics: low-cardinality Prometheus path labels to prevent time-series explosion on dynamic/404 paths.
- Cross-worker cache invalidation for the cached RAG service via a reload token in the data directory.

### Fixed
- FAISS persistence: ID map is now JSON (`id_map.json`) with atomic writes and best-effort locks; unsafe pickle maps are refused.
- API: request size limits for key endpoints to reduce DoS/cost-amplification risk.

## [0.1.1] - 2026-02-15

### Added
- LangChain loaders integration via `LangChainLoader` adapter implementing `LoaderPort`.
- Optional extras group `loaders` with `langchain-community` and `trafilatura` in `pyproject.toml`.
- Optional docs site scaffold (`mkdocs.yml` + `docs/index.md`).

### Fixed
- API: reset cached RAG service after ingestion; make readiness fail when dense index/id-map are missing.
- OpenAI: avoid embeddings calls for empty input; generator requires API key.
- FAISS: validate `id_map.json` format; guard ids/embeddings length mismatch.
- Settings: avoid side effects at import-time; create data dir at startup/scripts.

### Documentation
- Align package name and defaults (Ollama model, coverage instructions, config source of truth).
- Update architecture doc to match current ports/factory and list extra API endpoints.

### CI / Build
- Prefer Ruff (`ruff check` + `ruff format`) as the primary formatter/linter.
- CI: add `ruff format --check`, set `UV_CACHE_DIR`, align Docker tag with compose.
- Dockerfile: production stage now reuses the installed project from the deps stage (avoids rebuilding in the final image).

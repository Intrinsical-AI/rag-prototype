# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [Unreleased]

### Refactor
- Internal type/module taxonomy normalized by layer:
  - `core/services/schemas.py` replaced by `core/services/types.py` for data-only DTOs.
  - `OverlapV1Reranker` moved to `core/services/reranking.py` (behavioral strategy).
  - `app/services/results.py` added for docs mutation use-case outcomes.
  - `app/services/ports.py` kept focused on dependency contracts and bundles.
- Removed residual `core/services/models.py` (unused/empty).
- Internal imports updated across app/core/infra/CLI to align with new boundaries.

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

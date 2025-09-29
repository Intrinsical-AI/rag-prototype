# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]
- (nothing yet)

## [1.1.0] - 2025-09-28
### Added
- OpenRouter support with configurable settings (`OPENROUTER_ENABLED`, `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `OPENROUTER_MODEL`, `OPENROUTER_SITE_URL`, `OPENROUTER_APP_TITLE`).
- New endpoint `POST /api/openrouter/generate` (OpenAI-compatible proxy via OpenRouter).
- New endpoint `GET /api/templates` to retrieve available prompt templates.
- New endpoint `GET /api/config` to expose backend defaults and available providers.
- New endpoint `GET /api/health/ollama` to check Ollama server availability.
- Health and readiness endpoints hardened:
  - `GET /api/health` for DB liveness.
  - `GET /api/ready` checks DB, RAG service, LLM providers, and retrieval index if dense/hybrid.
- Metrics middleware and `GET /metrics` endpoint (Prometheus) behind `ENABLE_MONITORING` flag.
- IngestionPipeline orchestrator (preprocess → chunk → format → ETL) with `LoadedItem` support.
- Settings additions: `OPENAI_TOP_P`, `CSV_HAS_HEADER`.
- CI split into dedicated jobs with caching (Ubuntu-only test matrix): lint (ruff/black/isort), typing (mypy), tests (pytest+coverage), security (bandit+safety), docker_build (Buildx, cache-to/from gha).
- docker-compose with optional Ollama service and persistent volumes.
- Makefile developer targets: `lint`, `type`, `test`, `sec`, `docker-build`, `compose-up`, `compose-down`.
- Expanded test suite: parametrized edge cases for chunker, ETL, CSV loader, API validation; total tests now 125 with coverage ≥85%.

### Changed
- CLI `rag-status` formatting and output usability improvements.
- `bootstrap.py` updated to use `IngestionPipeline` and correct parameter name (`chunk_fn`).
- `ask_eval` routing corrected and simplified (single execution, correct decorator placement).

### Fixed
- RAG service returns a helpful message for empty retrieval results (and logs history with empty sources).
- FAISS dimension mismatch error message standardized to start with `FAISS dim mismatch`.
- OpenAI generator now imports settings and honors defaults; added `openai_top_p` setting.
- Multiple tests added to cover health/ready, OpenRouter, templates/config, dependencies, and middleware; total coverage ≥85%.

### Testing
- 100 tests passing across unit, integration, and e2e suites.
- Coverage: 85%.

[Unreleased]: https://github.com/Intrinsical-AI/rag-prototype/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/Intrinsical-AI/rag-prototype/releases/tag/v1.1.0

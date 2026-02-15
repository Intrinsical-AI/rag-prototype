# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [Unreleased]

### Added
- LangChain loaders integration via `LangChainLoader` adapter implementing `LoaderPort`.
- Optional extras group `loaders` with `langchain-community` and `trafilatura` in `pyproject.toml`.

### Tests
- Comprehensive unit tests for `LangChainLoader` covering attribute/doc mapping, dict fallback,
  generator support, drop-empty behavior, metadata filtering, stringify fallback,
  and ingestion pipeline integration.

### Documentation
- README section: "LangChain loaders integration (optional)" with install and usage examples.
- New doc `docs/langchain_loaders.md` with detailed instructions and troubleshooting.

### Chore / Build
- Updated `uv.lock` due to new optional extras.
- Ran `ruff`, `black`, and `isort` across the repo; committed resulting formatting/import changes.
- CI: added `ruff format --check`, set `UV_CACHE_DIR`, and aligned Docker image tag with compose.
- Tests/Coverage: scope coverage to `local_rag_backend` and stop generating HTML/XML reports by default (CI still uploads XML).
- Dockerfile: production stage now reuses the installed project from the deps stage (avoids rebuilding in the final image).
- Settings: keep `Settings` side-effect free (no mkdir on import); create the data dir at startup/scripts/bootstrap instead.
- Dev tooling: pre-commit mypy hook now includes `pydantic-settings`.

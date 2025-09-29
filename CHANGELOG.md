# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [Unreleased] - 2025-09-29

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

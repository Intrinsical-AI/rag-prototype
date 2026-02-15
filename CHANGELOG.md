# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [Unreleased]

_Nothing yet._

## [1.1.1] - 2026-02-15

### Added
- LangChain loaders integration via `LangChainLoader` adapter implementing `LoaderPort`.
- Optional extras group `loaders` with `langchain-community` and `trafilatura` in `pyproject.toml`.
- Optional docs site scaffold (`mkdocs.yml` + `docs/index.md`).

### Fixed
- API: reset cached RAG service after ingestion; make readiness fail when dense index/id-map are missing.
- OpenAI: avoid embeddings calls for empty input; generator requires API key.
- FAISS: validate `id_map.pkl` shape; guard ids/embeddings length mismatch.
- Settings: avoid side effects at import-time; create data dir at startup/scripts.

### Documentation
- Align package name and defaults (Ollama model, coverage instructions, config source of truth).
- Update architecture doc to match current ports/factory and list extra API endpoints.

### CI / Build
- Prefer Ruff (`ruff check` + `ruff format`) as the primary formatter/linter.
- CI: add `ruff format --check`, set `UV_CACHE_DIR`, align Docker tag with compose.
- Dockerfile: production stage now reuses the installed project from the deps stage (avoids rebuilding in the final image).

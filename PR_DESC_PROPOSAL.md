# PR Description Proposal

Base: `develop`
Head: `release/02-2026`

Diff summary: **65 files changed, 1504 insertions(+), 633 deletions(-)**.

## Why

This PR focuses on hardening the backend (API/service lifecycle, settings side-effects, persistence validation), improving CI/Docker reliability, and bringing docs in sync with the current implementation. It also adds an optional docs site scaffold (MkDocs) to match the existing `docs` extra.

## What Changed

### API / Runtime Behavior

- Readiness (`GET /api/ready`) is stricter for dense/hybrid:
  - returns **503** when the FAISS index or id-map files are missing (instead of a warning).
- Ingestion (`POST /api/docs`) resets the cached `RagService` so subsequent requests see newly ingested docs/index.
- Startup now ensures the configured `DATA_DIR` exists before SQLite is used.

### Settings / Side-Effects

- `Settings` is now side-effect free (no directory creation during import/instantiation).
- Directory creation is done explicitly at app startup and in scripts/bootstrap helpers.

### Providers / Embeddings / Vector Index

- OpenAI embedder:
  - no API call for empty input.
  - dimensionality fallback no longer depends on import-time settings.
- OpenAI generator:
  - fails fast when no API key is configured.
- FAISS index:
  - validates `id_map.pkl` format (must be `list[int]`).
  - guards ids/embeddings length mismatch.

### DevEx / Quality Gates

- CI:
  - adds `ruff format --check`.
  - sets `UV_CACHE_DIR=.uv-cache` to avoid non-writable HOME issues.
  - aligns Docker image tag with compose.
  - coverage output is standardized to `coverage.xml`.
- Makefile:
  - uses `ruff format --check` (not black/isort).
  - adds `make clean`.
- pre-commit:
  - mypy hook includes `pydantic-settings`.

### Docs

- README:
  - removes the Black badge (Ruff is the formatter).
  - fixes Ollama pull example to `gemma3:1b`.
  - adds a "Docs" section linking to `docs/*`.
  - clarifies "source of truth" for configuration and documents key variables.
  - corrects dense/hybrid flow: OpenAI embeddings when `OPENAI_API_KEY` is set, otherwise SentenceTransformers.
- `docs/architecture.md`:
  - ports snippet matches `src/local_rag_backend/core/ports/__init__.py`.
  - removes stale factory excerpt; references `src/local_rag_backend/app/factory.py` as source of truth.
  - documents additional endpoints (`/api/docs`, `/api/ask_eval`, `/api/openrouter/generate`, etc.).
- `docs/custom_usage_guide.md`:
  - removes unused import and uses `Path.mkdir` for `DATA_DIR`.
- Adds a minimal docs site scaffold:
  - `mkdocs.yml`
  - `docs/index.md`

## Notable Compatibility Notes

- If any downstream code relied on `Settings(data_dir=...)` creating directories automatically, it must now ensure the directory exists (the FastAPI app and bundled scripts already do).
- Readiness semantics changed (more strict) for dense/hybrid; this is intentional for deployability.

## Testing

- `pytest` (unit + integration + e2e)
- `ruff check .`
- `mypy .`

## Release Note

- `CHANGELOG.md` includes a `1.1.1` section dated **2026-02-15**.
- `pyproject.toml` still declares `version = "1.1.0"`.
  - Decision needed: bump to `1.1.1` in this PR, or keep changelog as "next release" only.

## Commit List (develop..HEAD)

1. 7fa6beb chore(changelog): cut 1.1.1 entry and reset Unreleased
2. c92ef86 docs: add docs index, align defaults, and scaffold mkdocs
3. 987af15 docs(architecture): sync ports/factory and document extra endpoints
4. c634242 chore: update changelog
5. 265f62f chore(cli): align prog name and improve status output
6. 45c5aae docs: align package name and Ollama defaults
7. c4bce3f chore(dev): align pre-commit and Makefile with ruff format and local uv cache
8. 45b34e7 chore(docker): reuse deps stage in production image and add .dockerignore
9. 7b8b5b8 chore(ci): add ruff format check, set UV_CACHE_DIR, and align docker tag
10. 9d2e872 chore(test): scope coverage to package and simplify defaults
11. 7208790 chore(packaging): ship py.typed marker and tighten Protocol stub
12. 650ecef fix(settings): avoid side effects and create data dir at startup
13. ff51f20 fix(st): reference dense-st extra in missing dependency message
14. c580f90 fix(faiss): validate id_map and guard ids/embeddings mismatch
15. 7ebd65a fix(openai): guard empty embeddings and require API key
16. f3b64c2 fix(api): reset cached service after ingest and tighten readiness
17. f77e9df format: apply ruff format
18. f674b8b docs: align env defaults and package name with current dist
19. f88e6a2 feat(dense): prefer OpenAI embeddings; split dense extras; tighten indexing tests
20. 867c3df refactor(app): make app.factory the real composition root
21. 519d6de fix(obs): make Prometheus middleware optional and correct CORS credentials
22. 89114d8 chore(make): run tasks via uv with local cache
23. e2f8e52 docs: fix advanced usage loader example; align factory snippet
24. d672e45 feat(optional): lazy-import faiss/st; add numpy fallback for faiss index
25. 3af5bb4 fix(retrieval): keep scores aligned with filtered docs
26. 14216fe refactor(api): async endpoints and dependencies; switch tests to httpx ASGI
27. e7fca95 build: make dense deps optional; document uv/.venv workflow


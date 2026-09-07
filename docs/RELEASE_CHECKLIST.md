# Release Checklist

> Purpose: local release notes and pre-tag checklist for `rag-prototype`.
>
> Keep this file updated before cutting a public Git tag or GitHub Release.

## Pre-Tag Checklist

1. Freeze the release candidate:
   - worktree is clean
   - exact `target_ref` is recorded
   - branch policy is explicit (`master`, `develop`, or another release branch)
2. Validate version metadata:
   - `pyproject.toml` version matches the intended public version
   - tag naming follows the chosen convention (`vX.Y.Z` or `X.Y.Z`)
   - GitHub Release target points at the same commit that was tested
3. Validate runtime bootstrap:
   - `config.yaml` exists or is created from `config.example.yaml`
   - `rag-bootstrap` succeeds from a fresh checkout/copy
   - repeated `rag-bootstrap` is idempotent for sample data
4. Validate server smoke:
   - `/healthz` returns 200
   - `/` returns 200
   - `/openapi.json` returns 200
   - `/readyz` behavior is understood:
     - 200 when an LLM provider is configured and backend state is ready
     - 503 with actionable message when no LLM provider is configured
5. Run gates from the frozen ref:
   - `pytest -q`
   - `ruff check src tests`
   - `ruff format --check src tests`
   - `mypy src`
   - `lint-imports`
   - `pre-commit run --all-files`
   - `make smoke-embedding-api-wheel`

## Release candidate: 3.0.0

The verified remote canonical branch is `master`. Deliver `v3.0.0` from the
accepted commit after its CI and installed-artifact checks pass. Existing tags
and published artifacts are immutable.

Breaking changes relative to 2.1.0:

- Move HTTP document ingestion from `POST /api/docs` to
  `POST /api/docs/ingest`.
- Move conversation import from `POST /api/docs/import` to
  `POST /api/docs/import-conversations`.
- Remove the `performance-cpu` extra; select the current documented extras.
- Require a boolean `debug` configuration value instead of historical string
  aliases.

The supported typed embedding integration and MCP entrypoints keep their current
contracts. Migrate HTTP clients and configuration directly to the documented
interfaces; no legacy aliases are retained. This release does not introduce a
new guarantee for migration of historical SQLite databases.

The release also includes the already-implemented canonical import diagnostics,
HTTP/frontend hardening, and rejection of incomplete Elasticsearch enumeration
before destructive scope operations. Existing regression suites cover those
behaviors. Record final gate results, commit, and artifact hashes with the release;
historical validation results below do not certify this candidate.

## Historical RC Notes: 2026-07-10

Planned release:

- Canonical promotion path: `develop` -> `main`.
- Package/tag version: `2.1.0` / `v2.1.0`.
- Historical `2.0.1` and `v1.3.0` tags remain untouched.
- `2.1.0` is intentionally forward-only because GitHub already published
  `2.0.1`, even though that tagged commit reports package version `1.3.0`.
- Release scope: strict `rag_ask` MCP input handling, explicit
  `RAG_CONFIG_PATH`, safe Makefile environment bootstrap, and the installed
  typed embedding integration API.

Validated locally before remote promotion:

- Full suite: 742 passed, 4 skipped, 87.41% branch coverage.
- Ruff, formatting, mypy, pre-commit/security hooks: passed.
- Import-linter: 5 contracts kept.
- Installed-wheel embedding API smoke: passed outside the checkout with no
  SentenceTransformers dependency.

Remote CI, Docker, Python 3.11, final `main` commit, and release artifact hashes
must be recorded after GitHub authentication is restored.

## RC Notes: 2026-04-27

Assumed RC:

- Branch: `develop`
- Commit: `908d6feaa53e7477bf88c7ddb1ab06c788782bbd`
- Package version: `1.3.0`
- Python support: `>=3.11,<3.13`

Reviewer verdict: `ship_with_notes`.

Validated local happy path:

- `uv venv .venv` used CPython 3.12.12.
- `uv sync --frozen --extra server` succeeded.
- `rag-bootstrap` succeeded.
- Resulting database contained 30 documents and 0 history entries.
- Re-running `rag-bootstrap` kept 30 documents.
- `rag-server` exposed `/healthz`, `/`, and `/openapi.json` successfully.
- `/readyz` returned 503 when no LLM provider was configured, with an actionable message.

Validated gates:

- `pytest -q`: 711 passed, 4 skipped, coverage 87.08%.
- `ruff check src tests`: passed.
- `ruff format --check src tests`: passed.
- `mypy src`: passed.
- `lint-imports`: 4 contracts kept.
- `pre-commit run --all-files`: passed in a temporary initialized copy.

Reviewer findings and handling:

- Addressed in this documentation pass:
  - Quickstart makes the `config.yaml` contract explicit.
  - Quickstart states that `/readyz` and `/api/ask` need an enabled LLM provider.
  - `make sync` / `make test` avoid `dense-st` by default; SentenceTransformers stays opt-in.
- Still open:
  - Release tags, GitHub Releases, package version, and default branch need reconciliation before the next public release.

Known scope limits:

- Docker Compose was not validated in this RC pass.
- External backends were not validated in this RC pass.
- The validated path was local `local_split` + sparse.

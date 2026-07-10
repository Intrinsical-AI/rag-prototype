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

# Contributing

## Prerequisites

- Python 3.11+ (CI validates 3.11 and 3.12)
- `uv` installed
- Docker (for image checks)

## Local setup

```bash
uv venv .venv
source .venv/bin/activate
uv sync --frozen --extra dev --extra test --extra lint
```

## Mandatory quality gates (CI parity)

Run these before opening/updating a PR:

```bash
uv run pre-commit run --all-files
UV_CACHE_DIR=.uv-cache uv run --active --no-sync ruff check src tests
UV_CACHE_DIR=.uv-cache uv run --active --no-sync pytest -q
```

Additional gates used in CI:

```bash
make lint
make type
make sec         # strict security gate (fails on findings)
```

Optional local security audit that does not block:

```bash
make sec-soft
```

## Docker expectations

CI builds the production image only:

```bash
docker build --target production .
```

Rules for reproducibility:

- Keep `uv.lock` up to date when dependencies change.
- Do not replace `uv sync --frozen` with non-locked installs in CI scripts.
- Keep runtime image minimal; build toolchain belongs only in build stages.

## Pull request checklist

- Changes are atomic and scoped.
- Tests/linters/security checks pass locally.
- Documentation is updated when behavior, CI, or operational flows change.
- PR description includes rationale, risk, and verification commands executed.

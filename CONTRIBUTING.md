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
UV_CACHE_DIR=.uv_cache uv run --active --no-sync ruff check src tests
UV_CACHE_DIR=.uv_cache uv run --active --no-sync pytest -q
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

## Proxy security expectations

If the service is exposed beyond localhost (cloud VM, k8s ingress, reverse proxy):

- Set `API_KEY` (recommended default for public/networked deployments).
- Ensure the edge proxy sanitizes and controls `X-Forwarded-For` and `Forwarded`.
- Do not trust client-supplied forwarding headers unless the proxy overwrites/normalizes them.

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

### Docs parity checklist (architecture/API changes)

- If ports/contracts changed, update `docs/architecture.md` and keep method signatures aligned.
- If API endpoints or payload semantics changed, update `README.md` API section.
- If operational behavior changed (CI gates, Docker stages, security defaults), update `README.md` and this guide.
- If a docs-only PR changes architecture/API/ops guidance, run local checks manually (CI is skipped for docs-only changes by workflow `paths-ignore`).

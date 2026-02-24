# Roadmap

## Current Snapshot
- Architecture is solid (hexagonal + clear adapter seams), but still has transitional overlap between `app/application` and `app/services`.
- Core HTTP/API and CLI flows are stable and well-tested.
- Main technical debt is structural simplification, not missing features.

## Near Term (Now -> Next 2 PRs)
1. Keep transport boundaries strict:
   - No infra imports in `app/routers`.
   - No transport/schema imports in `app/application`.
2. Remove CLI indirection legacy:
   - Eliminate `_hooks` dynamic bridge.
   - Use explicit runtime helpers in `cli_commands/runtime.py`.
3. Keep docs aligned with real structure (`app/http`, `app/application`, `app/container`).

## Mid Term
1. Consolidate app-layer use cases:
   - Gradually move orchestration to `app/application`.
   - Leave compatibility shims in `app/services` temporarily.
2. Reduce composition duplication:
   - Keep `AppContainer` as the single composition source.
   - Keep `factory` focused on app-context lifecycle/cache invalidation.
3. Centralize runtime error mapping in one HTTP boundary path.

## Long Term
1. Unify ingestion paths (sparse/dense) to remove duplicate control flow.
2. Split mixed type modules (e.g. `core/services/schemas.py`) by domain concern.
3. Continue tightening architecture tests to prevent regressions.

## Quality Gates (Every Refactor PR)
- `uv run ruff check .`
- `uv run mypy src`
- `uv run pytest` (or targeted subset for the changed area)

# Technical Debt Register

> Scope: validated hotspots in `core/services`, `core/use_cases`, and `cli_commands`
>
> Status: current as of 2026-03-19, based on code inspection and targeted unit tests

## Summary

The main technical debt is not lack of tests, but concentrated orchestration, dynamic contracts, and CLI/API drift around write flows.

Highest-value targets:

1. Mutation orchestration and saga execution.
2. CLI/API contract alignment for docs import/mutate.
3. Multi-store maintenance duplication.
4. Ingestion planner coupling to CLI output and weak item typing.
5. Evaluation flow cleanup, but only as a bounded consolidation task.

## Validated Hotspots

| Area | Debt | Evidence | Risk | Priority |
| --- | --- | --- | --- | --- |
| Mutation saga | One module owns profile validation, journal lifecycle, before-image, SQL apply, vector apply, rollback, recovery, and reconcile | [`src/local_rag_backend/core/use_cases/_mutation_saga_executor.py`](../src/local_rag_backend/core/use_cases/_mutation_saga_executor.py) | High blast radius for write-path changes | High |
| Mutation coordinator | Re-resolves profile, mixes execution strategy, batching, and settings fallbacks | [`src/local_rag_backend/core/use_cases/docs_mutation.py`](../src/local_rag_backend/core/use_cases/docs_mutation.py) | Coordination logic spreads across layers | High |
| Canonical import path | Scope-sync and vector deletion still rely on dynamic repo capabilities outside the coordinator | [`src/local_rag_backend/core/use_cases/docs_import_canonical.py`](../src/local_rag_backend/core/use_cases/docs_import_canonical.py) | Scope replace semantics remain fragile and partially duplicated | High |
| CLI/API contract drift | CLI JSON parsing is manual; HTTP uses typed Pydantic schemas; `replace_scope` defaults differ | [`src/local_rag_backend/cli_commands/docs/docs_import_canonical.py`](../src/local_rag_backend/cli_commands/docs/docs_import_canonical.py), [`src/local_rag_backend/http/schemas/docs.py`](../src/local_rag_backend/http/schemas/docs.py) | Same business action behaves differently by transport | High |
| Maintenance | Two multi-store delete flows are near-mirror implementations | [`src/local_rag_backend/core/services/maintenance.py`](../src/local_rag_backend/core/services/maintenance.py) | Partial fixes and telemetry drift | Medium |
| Ingestion planner | Planning, stale detection, batching, mutation execution, and `click.echo` live in one module; `items` uses `Any` | [`src/local_rag_backend/cli_commands/docs/_ingestion_planner.py`](../src/local_rag_backend/cli_commands/docs/_ingestion_planner.py) | Reuse is limited; contracts are implicit | Medium |
| Evaluation | Dataset parsing is centralized, but flags and runtime wiring still span use case and CLI; compat args remain in core service | [`src/local_rag_backend/core/services/evaluation.py`](../src/local_rag_backend/core/services/evaluation.py), [`src/local_rag_backend/core/use_cases/evaluation.py`](../src/local_rag_backend/core/use_cases/evaluation.py), [`src/local_rag_backend/cli_commands/eval.py`](../src/local_rag_backend/cli_commands/eval.py) | Incremental options can drift across layers | Medium |

## Detailed Findings

### 1. Mutation orchestration is the main hotspot

Validated points:

- [`_mutation_saga_executor.py`](../src/local_rag_backend/core/use_cases/_mutation_saga_executor.py) is 665 lines and carries the core durable write flow.
- It uses dynamic access in critical paths: `getattr`, `hasattr`, `Any`, and `cast()` around settings, repo capabilities, journal state, and vector delta application.
- [`docs_mutation.py`](../src/local_rag_backend/core/use_cases/docs_mutation.py) resolves the storage profile twice and still owns strategy decisions that are larger than a thin coordinator.

Nuance:

- Some dynamic behavior is intentional because the port design allows optional adapter capabilities in [`core/ports/contracts.py`](../src/local_rag_backend/core/ports/contracts.py).
- That nuance does not justify the repeated `getattr(settings_obj, ...)` usage, because [`Settings`](../src/local_rag_backend/settings.py) already defines these fields.

Recommended direction:

- Extract a typed mutation runtime/config resolver.
- Split saga internals into explicit phases: profile checks, journal lifecycle, SQL phase, vector phase, recovery.
- Keep behavior stable first; do not combine this with feature work.

### 2. Canonical import and transport semantics are misaligned

Validated points:

- CLI defaults `replace_scope` to `True` when omitted in [`docs_import_canonical.py`](../src/local_rag_backend/cli_commands/docs/docs_import_canonical.py).
- HTTP defaults `replace_scope` to `False` in [`http/schemas/docs.py`](../src/local_rag_backend/http/schemas/docs.py).
- Scope replacement in [`core/use_cases/docs_import_canonical.py`](../src/local_rag_backend/core/use_cases/docs_import_canonical.py) still uses dynamic repo methods such as `list_external_ids_by_scope`, `snapshot_by_external_ids`, and `hard_delete_by_external_ids`.

Why this matters:

- This is not only technical debt; it is behavior drift across transports.
- The same payload shape can produce different deletion semantics depending on whether it comes from CLI or HTTP.

Recommended direction:

- Align default semantics across transports first.
- Introduce shared request validation for CLI payloads instead of hand-built dict parsing.
- Decide whether scope replacement belongs inside the canonical mutation flow or in a separate explicit use case.

### 3. Maintenance has cheap-to-fix duplication

Validated points:

- [`delete_documents_multi_store`](../src/local_rag_backend/core/services/maintenance.py) and [`delete_external_ids_multi_store`](../src/local_rag_backend/core/services/maintenance.py) repeat the same preflight, delete, fallback embedder, rebuild, and consistency-error pattern.

Recommended direction:

- Extract one private helper for common multi-store delete orchestration.
- Preserve the two public result DTOs, because their caller-facing semantics differ.

### 4. Ingestion planner is reusable in theory, but still CLI-shaped

Validated points:

- [`_ingestion_planner.py`](../src/local_rag_backend/cli_commands/docs/_ingestion_planner.py) combines file planning, stale detection, dedup, mutation execution, and terminal output.
- `IngestPlan.items` and multiple helpers use `Any` even though the items require a small stable surface: `external_id`, `content`, `source_id`, `metadata`.

Recommended direction:

- Separate pure planning from batch execution.
- Move user-facing `click.echo` to [`docs_ingest.py`](../src/local_rag_backend/cli_commands/docs/docs_ingest.py).
- Replace `Any` with a minimal `Protocol` or typed DTO.

### 5. Evaluation needs bounded cleanup, not a broad rewrite

Validated points:

- Dataset parsing is already centralized in [`load_eval_dataset`](../src/local_rag_backend/core/services/evaluation.py).
- JSON formatting and compare serialization are also centralized in the same service module.
- The real debt is option propagation and legacy compatibility, visible in the retained kwargs in [`run_retrieval_eval`](../src/local_rag_backend/core/services/evaluation.py).

Recommended direction:

- Keep dataset parsing and metrics pure in the service layer.
- Consolidate request/config DTOs across CLI and use case.
- Remove legacy compatibility parameters once all callers are migrated.

## What Is Not the Problem

These hotspots are not untested.

Relevant coverage exists in:

- Mutation flow and recovery: [`tests/unit/core/use_cases/test_docs_mutation_refactor.py`](../tests/unit/core/use_cases/test_docs_mutation_refactor.py)
- Maintenance consistency paths: [`tests/unit/core/services/test_maintenance.py`](../tests/unit/core/services/test_maintenance.py)
- Evaluation semantics and compare flow: [`tests/unit/core/services/test_evaluation.py`](../tests/unit/core/services/test_evaluation.py), [`tests/unit/application/services/test_app_evaluation_service.py`](../tests/unit/application/services/test_app_evaluation_service.py), [`tests/unit/cli/test_cli_eval.py`](../tests/unit/cli/test_cli_eval.py)
- CLI mutation boundaries: [`tests/unit/cli/test_cli_mutation_boundary.py`](../tests/unit/cli/test_cli_mutation_boundary.py)

Interpretation:

- The debt is primarily cognitive complexity and contract fragility, not absence of safety nets.
- Refactors should preserve behavior and test shape before attempting simplification.

## Prioritized Plan

### P0

- Align `replace_scope` defaults between CLI and HTTP.
- Introduce shared typed validation for CLI mutation and canonical-import payloads.

### P1

- Extract a typed mutation runtime/config object from `Settings`.
- Split mutation saga internals by phase without changing external behavior.
- Remove duplicate profile resolution from `MutationCoordinator`.

### P2

- Refactor `maintenance.py` with a shared helper.
- Decouple `_ingestion_planner.py` from terminal output and replace `Any` item contracts.

### P3

- Consolidate evaluation request/config handling.
- Remove legacy compatibility args once callers are normalized.

## Guardrails For Refactoring

- Do not rewrite the mutation stack and canonical import flow in one change.
- Preserve journal and recovery semantics while extracting helpers.
- Treat transport alignment as a behavior change with explicit tests.
- Keep HTTP schemas and CLI DTOs close enough that one cannot silently diverge.

## Validation Notes

The hotspots above were checked against current code and targeted tests.

Targeted test command used during review:

```bash
DEBUG=false UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/unit/core/use_cases/test_docs_mutation_refactor.py \
  tests/unit/core/services/test_maintenance.py \
  tests/unit/core/services/test_evaluation.py \
  tests/unit/application/services/test_app_evaluation_service.py \
  tests/unit/cli/test_cli_eval.py \
  tests/unit/cli/test_cli_mutation_boundary.py
```

Result:

- 60 tests passed.
- The command still exited non-zero because the repository enforces a global coverage threshold, not because these targeted tests failed.

# Technical Debt Register

> Scope: validated hotspots in `core/services`, `core/use_cases`, `cli_commands`, plus selected repo-level seams in `composition` and `infrastructure` when they materially affect runtime behavior
>
> Status: current as of 2026-04-08, based on code inspection and targeted unit tests

## Summary

The main debt is still concentrated orchestration and contract fragility around stateful write flows.

What changed since the previous revision:

1. Some earlier structural debt is now materially reduced:
   - runtime config is YAML-only and validated centrally in `Settings`
   - agent-facing status now uses a typed runtime snapshot instead of raw settings reads
   - canonical import transport now shares one typed validation/assembly path across CLI, HTTP, and MCP
   - `core/use_cases -> infrastructure|composition` import debt is frozen at an explicit zero baseline
   - advanced docs now teach canonical mutation instead of direct storage writes
2. The evaluation stack should no longer be treated as "bounded cleanup only".
   - It is tested and structurally cleaner than before
   - but it still has methodological blind spots that can hide retrieval defects or overstate candidate quality

Highest-value active targets:

1. Mutation orchestration and saga execution.
2. CLI/API contract alignment for canonical import.
3. Evaluation methodology and ranking fidelity.
4. CLI / DX contract consistency.
5. Multi-store maintenance duplication.
6. Ingestion planner typing and CLI coupling.

## Resolved Or Materially Reduced Debt

These items should not continue to be treated as primary debt drivers:

- Runtime configuration source drift:
  `Settings` is now a YAML-validated `BaseModel`, not env-driven `BaseSettings`.
  Evidence: [`src/local_rag_backend/settings.py`](../src/local_rag_backend/settings.py)
- Use-case import boundary drift:
  the architecture test now enforces a zero-debt baseline for `core/use_cases -> infrastructure|composition`.
  Evidence: [`tests/architecture/test_architecture_use_case_infra_debt.py`](../tests/architecture/test_architecture_use_case_infra_debt.py)
- Canonical write-path examples in docs:
  advanced usage now shows `AppContainer + MutationCoordinator`, not direct SQL writes.
  Evidence: [`docs/USAGE.md`](./USAGE.md)

Interpretation:

- The repo is no longer carrying generic "config sprawl" debt as a top concern.
- The remaining debt is more specific: orchestration complexity, transport drift, and evaluation rigor.

## Active Hotspots

| Area | Debt | Evidence | Risk | Priority |
| --- | --- | --- | --- | --- |
| Mutation saga | One module still owns profile validation, journal lifecycle, before-image, SQL apply, vector apply, rollback, recovery, and reconcile | [`src/local_rag_backend/core/use_cases/_mutation_saga_executor.py`](../src/local_rag_backend/core/use_cases/_mutation_saga_executor.py) | High blast radius for write-path changes | High |
| Mutation coordinator | Strategy selection, profile resolution, batching, and settings-derived behavior remain concentrated in one coordinator | [`src/local_rag_backend/core/use_cases/docs_mutation.py`](../src/local_rag_backend/core/use_cases/docs_mutation.py) | Coordination logic still spreads across layers | High |
| Canonical import path | Scope-sync still relies on dynamic repo capabilities outside the coordinator; transport validation is now shared, but the write semantics still depend on repo-specific delete hooks | [`src/local_rag_backend/core/use_cases/docs_import_canonical.py`](../src/local_rag_backend/core/use_cases/docs_import_canonical.py), [`src/local_rag_backend/cli_commands/docs/docs_import_canonical.py`](../src/local_rag_backend/cli_commands/docs/docs_import_canonical.py) | Same business action can still depend on repo capabilities outside the core coordinator | High |
| Evaluation methodology | Unknown retrieved IDs are silently filtered; original retriever scores are discarded; compare gate uses aggregate deltas only | [`src/local_rag_backend/core/services/evaluation.py`](../src/local_rag_backend/core/services/evaluation.py), [`src/local_rag_backend/core/use_cases/evaluation.py`](../src/local_rag_backend/core/use_cases/evaluation.py) | Retrieval defects can be masked; candidate quality can be overstated | High |
| CLI / DX contract consistency | Evaluation flags are still powerful and cognitively expensive; some command surfaces remain manually shaped rather than spec-driven | [`src/local_rag_backend/cli_commands/docs/docs_mutate.py`](../src/local_rag_backend/cli_commands/docs/docs_mutate.py), [`src/local_rag_backend/cli_commands/eval.py`](../src/local_rag_backend/cli_commands/eval.py) | Users still have to learn a wide CLI surface | Medium |
| Maintenance | Two multi-store delete flows are still near-mirror implementations | [`src/local_rag_backend/core/services/maintenance.py`](../src/local_rag_backend/core/services/maintenance.py) | Partial fixes and telemetry drift | Medium |
| Ingestion planner | Planning, stale detection, batching, mutation execution, and terminal output still live in one module; `items` remains `Any` | [`src/local_rag_backend/cli_commands/docs/_ingestion_planner.py`](../src/local_rag_backend/cli_commands/docs/_ingestion_planner.py) | Reuse is limited; contracts remain implicit | Medium |
| Elasticsearch system state | `bump_version()` is still read-modify-write without an atomic compare-and-swap or conflict retry loop | [`src/local_rag_backend/infrastructure/persistence/elasticsearch/system_state.py`](../src/local_rag_backend/infrastructure/persistence/elasticsearch/system_state.py) | Cross-worker cache invalidation can lose increments under contention | Medium |

## Detailed Findings

### 1. Mutation orchestration remains the highest-risk hotspot

Validated points:

- [`_mutation_saga_executor.py`](../src/local_rag_backend/core/use_cases/_mutation_saga_executor.py) is still 665 lines and carries the durable write flow.
- [`docs_mutation.py`](../src/local_rag_backend/core/use_cases/docs_mutation.py) still resolves storage profile and execution mode before delegating into saga or atomic execution.
- Critical paths still depend on optional adapter capabilities and dynamic checks around journal, vector delta, and rollback support.

Nuance:

- Some dynamic behavior is intentional because `DocsMutationPorts` supports optional capabilities.
- That does not eliminate the debt: the hot path is still hard to reason about, hard to refactor safely, and expensive to extend.

Recommended direction:

- Extract a typed mutation runtime/config resolver from `Settings`.
- Split saga internals into explicit phases: profile checks, journal lifecycle, SQL phase, vector phase, recovery.
- Keep behavior stable first; do not combine this with transport or feature work.

### 2. Canonical import semantics are still transport-fragile

Validated points:

- CLI and HTTP now default `replace_scope` to `True` when omitted in canonical import.
- Scope replacement in [`docs_import_canonical.py`](../src/local_rag_backend/core/use_cases/docs_import_canonical.py) still depends on dynamic repo methods such as `list_external_ids_by_scope`, `snapshot_by_external_ids`, and `hard_delete_by_external_ids`.

Why this matters:

- This is both debt and behavior drift.
- The same business action can still produce different deletion semantics depending on transport.
- Scope replace is not fully absorbed into the canonical mutation flow; it still leans on repo-specific escape hatches.

Recommended direction:

- Keep the shared `replace_scope=true` default stable across transports.
- Introduce shared typed validation for CLI payloads instead of hand-built dict parsing.
- Decide whether scope replacement belongs inside canonical mutation or deserves a separate explicit use case.

### 3. Evaluation is structurally cleaner, but methodologically underpowered

Validated points:

- Dataset parsing and aggregate metric calculation are correctly centralized in [`load_eval_dataset`](../src/local_rag_backend/core/services/evaluation.py) and `ir_measures`.
- The evaluation workspace is isolated through [`prepare_eval_workspace`](../src/local_rag_backend/core/use_cases/evaluation.py), so this is not a simple "production index accidentally reused" story.
- The core evaluator still filters retrieved IDs not present in the dataset corpus:
  `normalized_external_id not in known_doc_ids` in [`run_retrieval_eval`](../src/local_rag_backend/core/services/evaluation.py).
- The evaluator also discards original retrieval scores because the callback contract is `Sequence[str]`, then invents rank-based scores via `k - rank + 1`.
- [`compare_eval_results`](../src/local_rag_backend/core/services/evaluation.py) still gates on aggregate deltas only, with no per-query distribution or significance testing.

Why this matters:

- Filtering unknown IDs can hide retrieval-state defects or corpus leakage instead of surfacing them as degraded precision.
- Rank-only callback contracts make the evaluator lossy and block better future analysis.
- Aggregate-only gates are acceptable as operational guardrails, but not as evidence of statistical superiority.

Nuance:

- The current design is not "wrong" for a simple regression gate.
- It is wrong to treat it as a complete retrieval evaluation methodology.

Recommended direction:

- Change the retrieval callback contract to return `(external_id, score)` pairs.
- Stop silently filtering unknown IDs; either keep them as non-relevant results or surface them as explicit evaluator anomalies.
- Extend the dataset format to support optional graded relevance.
- Add per-query outputs and, later, paired significance testing for compare mode.

### 4. CLI / DX contract consistency is now first-order debt

Validated points:

- CLI mutation and canonical-import commands still parse JSON manually instead of reusing shared typed validation.
- `replace_scope` semantics are aligned and the typed validation path is now shared across CLI, HTTP, and MCP.
- `rag-eval`, `rag-eval-batch`, and `rag-eval-compare` expose powerful workflows, but the option surface is large and uneven:
  flags such as `--candidate-candidate-k`, `--baseline-dual-candidate-k`, and JSON batch specs create a high cognitive load.
- `_ingestion_planner.py` still emits terminal output directly, which keeps planning logic coupled to CLI behavior.

Why this matters:

- This is not cosmetic UX debt; it is contract drift.
- A user can learn one surface in CLI and hit different semantics in HTTP.
- Refactors become riskier because parsing, defaults, and messages are duplicated across commands.

Recommended direction:

- Reuse DTOs / use-case input models for CLI payload validation.
- Align visible defaults across CLI and HTTP.
- Simplify evaluation entrypoints with profiles or spec-driven compare flows, not only flag expansion.
- Standardize exit codes and success/error output shape across commands.

### 5. Maintenance still has cheap-to-fix duplication

Validated points:

- [`delete_documents_multi_store`](../src/local_rag_backend/core/services/maintenance.py) and [`delete_external_ids_multi_store`](../src/local_rag_backend/core/services/maintenance.py) still repeat preflight, delete, embedder fallback, rebuild, and consistency-error handling.

Recommended direction:

- Extract one private helper for common multi-store delete orchestration.
- Preserve the two public result DTOs because caller-visible semantics still differ.

### 6. Ingestion planner is still CLI-shaped and weakly typed

Validated points:

- [`_ingestion_planner.py`](../src/local_rag_backend/cli_commands/docs/_ingestion_planner.py) still combines file planning, stale detection, dedup, mutation execution, and `click.echo`.
- `IngestPlan.items` remains `tuple[Any, ...]`, even though the required surface is small and stable (`external_id`, `content`, `source_id`, `metadata`).

Recommended direction:

- Separate pure planning from batch execution.
- Move terminal output back to [`docs_ingest.py`](../src/local_rag_backend/cli_commands/docs/docs_ingest.py).
- Replace `Any` with a small Protocol or DTO.

### 6. Elasticsearch system state is a real concurrency blind spot

Validated points:

- [`ElasticSystemStateStorage.bump_version()`](../src/local_rag_backend/infrastructure/persistence/elasticsearch/system_state.py) reads the current version and writes `current + 1` back without compare-and-swap or conflict retry.
- This storage is used to invalidate cached runtime state from `AppContainer`.

Why this matters:

- Under concurrent writers, version increments can be lost.
- That means cache invalidation is best-effort, not monotonic.

Recommended direction:

- Replace read-modify-write with an atomic Elasticsearch update strategy.
- Add a contention-focused test, not only a monotonic sequential test.

## What Is Not The Problem

These hotspots are not primarily about missing tests.

Relevant coverage exists in:

- Mutation flow and recovery: [`tests/unit/core/use_cases/test_docs_mutation_refactor.py`](../tests/unit/core/use_cases/test_docs_mutation_refactor.py)
- Maintenance consistency paths: [`tests/unit/core/services/test_maintenance.py`](../tests/unit/core/services/test_maintenance.py)
- Evaluation semantics and compare flow: [`tests/unit/core/services/test_evaluation.py`](../tests/unit/core/services/test_evaluation.py), [`tests/unit/application/services/test_app_evaluation_service.py`](../tests/unit/application/services/test_app_evaluation_service.py), [`tests/unit/cli/test_cli_eval.py`](../tests/unit/cli/test_cli_eval.py)
- Mutation port wiring: [`tests/unit/application/services/test_mutation_ports.py`](../tests/unit/application/services/test_mutation_ports.py)
- CLI mutation boundaries: [`tests/unit/cli/test_cli_mutation_boundary.py`](../tests/unit/cli/test_cli_mutation_boundary.py)
- Use-case import boundary freeze: [`tests/architecture/test_architecture_use_case_infra_debt.py`](../tests/architecture/test_architecture_use_case_infra_debt.py)

Interpretation:

- The remaining debt is primarily cognitive complexity, contract fragility, and methodological blind spots.
- Refactors should preserve behavior and test shape before attempting simplification.

## Prioritized Plan

### P0

- Reuse one shared typed validation path between CLI and HTTP.
- Introduce shared typed validation for CLI canonical-import and mutation payloads.
- Fix evaluation blind spots:
  - stop silently filtering unknown IDs
  - preserve original retriever scores in the eval callback contract
- Start CLI / DX normalization:
  - unify validation for `mutate-docs`, `import-canonical`, and `eval-batch`
  - align visible defaults across CLI and HTTP

### P1

- Extract a typed mutation runtime/config object from `Settings`.
- Split mutation saga internals by phase without changing external behavior.
- Remove duplicate profile resolution and strategy branching where possible from `MutationCoordinator`.
- Add per-query outputs to evaluation compare mode.
- Reduce `eval-compare` flag complexity with profiles or spec-file support.
- Homogenize CLI exit codes and success/error output shape.

### P2

- Refactor `maintenance.py` with a shared helper.
- Decouple `_ingestion_planner.py` from terminal output and replace `Any` item contracts.
- Harden `ElasticSystemStateStorage.bump_version()` with atomic update semantics.
- Review evaluation help texts and flag naming for consistency.

### P3

- Extend eval dataset schema to support optional graded relevance.
- Add paired significance testing for compare mode once per-query outputs exist.

## Guardrails For Refactoring

- Do not rewrite the mutation stack and canonical import flow in one change.
- Preserve journal and recovery semantics while extracting helpers.
- Treat transport-alignment work as explicit behavior change with tests.
- Keep HTTP schemas and CLI DTOs close enough that one cannot silently diverge.
- Do not present aggregate-delta compare gates as statistical significance.

## Validation Notes

The hotspots above were checked against current code and targeted tests.

## Quick Wins (next 2-4 weeks)

This section prioritizes low-risk, high-impact improvements for the three
highest-impact active debts.

### A) Mutation saga monolítica

1. **Extract phase helpers without changing orchestration order**
   - Create private functions for: preflight/profile checks, SQL apply, vector
     apply, rollback/recovery.
   - **Why:** Reduces cognitive load and blast radius while preserving current
     write semantics.
2. **Introduce a small typed runtime config object**
   - Resolve toggles from `Settings` once and pass a typed object through the
     mutation path.
   - **Why:** Removes repeated dynamic branching and makes behavior easier to
     reason about and test.
3. **Add phase-scoped structured telemetry**
   - Emit consistent phase markers (`phase`, `doc_count`, `strategy`) around saga
     steps.
   - **Why:** Makes production debugging faster before deeper refactors.

### B) Evaluación con sesgos metodológicos

1. **Preserve retrieval scores in the callback contract**
   - Move from `Sequence[str]` to a scored shape (`external_id`, `score`).
   - **Why:** Avoids lossy rank-only evaluation and unlocks better diagnostics.
2. **Stop silently dropping unknown IDs**
   - Treat unknown IDs as explicit anomalies (counter + optional report) or as
     non-relevant retrieved docs.
   - **Why:** Prevents false confidence by surfacing corpus/index drift.
3. **Emit per-query breakdown in compare mode**
   - Add query-level metrics output alongside aggregate deltas.
   - **Why:** Quick visibility into regressions hidden by averages.

### C) Fragilidad de contratos CLI/HTTP

1. **Single DTO validation path for CLI + HTTP + MCP**
   - Reuse the same typed input models in all transports.
   - **Why:** Eliminates semantic drift and duplicated parsing logic.
2. **Unify visible defaults and error shape**
   - Align default values and error payload contract across entrypoints.
   - **Why:** Reduces user confusion and makes integration tests more stable.
3. **Add transport parity tests for critical commands**
   - Golden tests asserting same payload => same behavior in CLI and HTTP.
   - **Why:** Catches drift early with low implementation cost.

## Suggested execution order

1. **Week 1-2:** transport parity + shared DTO validation (fast risk reduction).
2. **Week 2-3:** evaluator unknown-ID handling + scored callback contract.
3. **Week 3-4:** mutation saga phase extraction (behavior-preserving slice).

This order is recommended because it first reduces outward-facing contract
fragility, then improves quality signals, and finally tackles internal
orchestration complexity with lower operational risk.

Targeted test command used during this review:

```bash
uv run pytest -q -o addopts='' \
  tests/unit/core/use_cases/test_docs_mutation_refactor.py \
  tests/unit/core/services/test_maintenance.py \
  tests/unit/core/services/test_evaluation.py \
  tests/unit/application/services/test_mutation_ports.py \
  tests/unit/cli/test_cli_mutation_boundary.py \
  tests/unit/http/test_docs_index_ports_wiring.py
```

Result:

- 50 tests passed.

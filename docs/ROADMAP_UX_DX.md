# UX / DX Roadmap

> Scope: operator and developer ergonomics for CLI and adjacent transport contracts
>
> Status: current as of 2026-04-08

This roadmap isolates the UX / DX work so it can move independently from mutation-engine and evaluation-core refactors.

## P0

- Unify CLI payload validation with DTOs / use cases for:
  - `rag-mutate-docs`
  - `rag-import-canonical`
  - `rag-eval-batch`
- Align visible defaults between CLI and HTTP:
  - especially `replace_scope`
  - avoid one transport being destructive-by-default while the other is not

## P1

- [x] Reduce `rag-eval-compare` complexity:
  - use a canonical `--spec` file with baseline, candidate, and thresholds
  - remove expanded baseline/candidate flag matrices
- Homogenize CLI exit codes and success/error output:
  - consistent distinction between contract errors, runtime errors, and failed evaluation gates
  - consistent success summaries across commands

## P2

- Remove `click.echo` from [`src/local_rag_backend/cli_commands/docs/_ingestion_planner.py`](../src/local_rag_backend/cli_commands/docs/_ingestion_planner.py) so planning stays reusable and transport-neutral
- Review evaluation help texts and flag naming:
  - reduce duplicated mental models around `candidate_k`, `dual_candidate_k`, `hybrid_alpha`
  - make baseline/candidate option names easier to scan

## Guardrails

- Do not redesign CLI UX in the same change as mutation-engine refactors.
- Treat defaults alignment as behavior change and cover it with tests.
- Keep CLI and HTTP semantics close enough that docs do not have to explain transport-specific caveats.

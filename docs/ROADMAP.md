# General Roadmap

> Scope: prioritized delivery roadmap derived from the current technical-debt register and the current codebase
>
> Status: current as of 2026-04-08

This roadmap is execution-oriented and intentionally atomic. Each item should be implementable and reviewable on its own, without bundling large architectural rewrites into one change.

## P0

- `transport`: align canonical import semantics between CLI and HTTP
  - unify `replace_scope` defaults
  - cover the behavior explicitly in transport tests
- `eval`: preserve retriever scores in offline evaluation
  - change the eval callback contract from `Sequence[str]` to `(external_id, score)` pairs
- `eval`: stop silently filtering unknown retrieved IDs
  - either keep them as non-relevant results or emit explicit evaluator anomalies
- `ux/dx`: unify CLI payload validation with DTOs / use cases for:
  - `rag-mutate-docs`
  - `rag-import-canonical`
  - `rag-eval-batch`

## P1

- `mutation`: extract a typed mutation runtime/config object from `Settings`
- `mutation`: remove duplicated profile resolution and strategy branching from `MutationCoordinator`
- `mutation`: split `_mutation_saga_executor.py` into explicit phases without changing behavior
- `eval`: emit per-query outputs for compare mode
  - this is the prerequisite for stronger statistical comparison later
- `ux/dx`: reduce `rag-eval-compare` complexity with profiles or spec-file support

## P2

- `maintenance`: extract a shared helper for multi-store delete orchestration
- `ingest`: replace `Any` item contracts in `_ingestion_planner.py` with a small Protocol or DTO
- `ingest`: remove `click.echo` from planner internals so planning stays transport-neutral
- `state`: make `ElasticSystemStateStorage.bump_version()` atomic
  - add a contention-focused test, not only a sequential monotonic test
- `ux/dx`: homogenize CLI exit codes and success/error output shape

## P3

- `eval`: extend dataset schema to support optional graded relevance
- `eval`: add paired significance testing for compare mode once per-query outputs exist
- `canonical import`: decide whether scope replacement belongs inside canonical mutation or a dedicated use case
- `ux/dx`: review evaluation help texts and flag naming for consistency and scanability

## Guardrails

- Do not combine mutation-engine refactors with transport-semantic changes in the same PR.
- Treat defaults alignment as behavior change and cover it with tests.
- Preserve journal and recovery semantics while extracting mutation helpers.
- Do not present aggregate-delta compare gates as statistical significance.

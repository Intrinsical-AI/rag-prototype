# ADR-006: Mutation UnitOfWork With Shared SQL Session

## Status
Accepted and implemented

## Date
2026-03-01

## Context / Problema
`MutationCoordinator` introduced `mutation_uow_factory`, but SQL repositories still committed internally per method call. This limited transactional control and made rollback/commit boundaries less explicit from application orchestration.

Symptoms:
- SQL write methods committed eagerly even under an application-level UoW.
- SQL compensation paths were not guaranteed to run under the same explicit transaction boundary mechanism.
- Tests could not clearly prove commit/rollback behavior driven by application orchestration.

## Alternatives considered
1. Keep per-method commits in repositories and treat UoW as logical-only.
- Rejected: weak transactional contract, less predictable failure semantics.

2. Rewrite all SQL repositories and services around a full explicit Session injection API.
- Rejected for now: larger refactor blast radius than needed for deliverable stabilization.

3. Add a shared-session UoW context and make repositories opt into it transparently.
- Accepted: minimizes API churn while providing explicit transactional boundaries where needed.

## Decision
We implemented a shared-session UoW model for mutation paths:
1. Add `session_uow()` and `get_bound_session()` in `sql/base.py`.
2. Extend `DocsMutationPorts` with `mutation_uow_factory`.
3. Wire default mutation UoW to `session_uow` in composition.
4. Update `SqlDocumentStorage` and `HistorySqlStorage` to:
- reuse bound session when present;
- avoid internal early commit under shared UoW;
- flush instead of commit when session is externally owned.
  (implemented through `get_managed_session()` + `autocommit` flags in SQL CRUD adapters)
5. Keep fallback behavior unchanged when no UoW is active (own session + commit).

## Consequences
Positive:
- Application layer now controls SQL transaction boundaries for canonical mutation and SQL compensation paths.
- Commit/rollback semantics are explicit and testable.
- No external API/CLI contract changes were required.

Negative:
- Added complexity in SQL adapter session management.
- Transaction guarantees are stronger in mutation paths, but not yet uniformly applied to every write workflow.

## Safeguards / Tests
- `tests/unit/infrastructure/persistence/sql/test_sql_storage.py`
  - verifies shared-UoW commit behavior
  - verifies shared-UoW rollback behavior
- `tests/unit/core/use_cases/test_docs_mutation_refactor.py`
  - verifies UoW wrapping on SQL mutation path and rollback compensation path

## Follow-ups
- [TODO: Fase C] Evaluate extending shared-UoW transactional seams to other write-heavy workflows beyond canonical mutation.
- Keep avoiding direct SQL writes outside application-orchestrated boundaries.

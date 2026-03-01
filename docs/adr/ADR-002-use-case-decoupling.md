# ADR-002: Use-Case Decoupling Guardrails

## Status
Accepted and implemented

## Date
2026-03-01

## Context / Problema
`core/use_cases` imported concrete infrastructure adapters in multiple modules. This violated the intended architecture boundary and increased change coupling, testing friction, and regression risk.

Final state:
- `core/use_cases` no longer imports `infrastructure` or `composition`.
- All target use cases are wired through ports in `core/ports/use_cases.py`.
- Architecture debt snapshot for `core/use_cases -> infrastructure|composition` is empty.
- Follow-up internal refactors (B2.2/B2.3) keep the rule sustainable: mutation coordinator helpers extracted and HTTP/CLI wiring fan-out reduced via container context bundles/builders.

## Alternatives considered
1. Keep soft architecture rules in docs only.
- Rejected: no enforceable gate; debt can grow unnoticed.

2. Block all existing violations immediately.
- Rejected initially for Fase A: would force large refactor before contracts/ports were prepared.

3. Freeze current debt snapshot and block any new debt.
- Accepted: allows incremental refactor while preventing boundary regressions.

## Decision
We completed the migration and now enforce a strict rule:
1. `core/use_cases -> infrastructure|composition` imports are forbidden.
2. Snapshot exceptions are removed (empty baseline).
3. New use-case integrations must be wired through ports/adapters only.

## Consequences
Positive:
- Architectural drift is prevented immediately.
- CI boundary failures are explicit and immediate.
- Use-case logic is transport/infrastructure agnostic.
- Follow-up transactional hardening is possible without reintroducing infra imports in use cases (see ADR-006).

Negative:
- Port wiring introduces additional composition code.

## Debt burn-down policy
- Snapshot is empty and remains empty.
- Any new exception requires explicit ADR update and technical lead approval.

## Review policy
- Boundary test failures are treated as architecture regressions.
- Adding `core/use_cases -> infrastructure|composition` imports is not allowed by default.

Target removal date:
- Completed on 2026-03-01.

## Fase A Exit Checklist
- [x] Guardrails active in architecture tests.
- [x] Frozen debt snapshot reduced to empty baseline.
- [x] Target use-case ports defined and exported.
- [x] CI `architecture` job active.

# ADR-007: Persistence Topology Abstraction (Split-Store and Unified-Store)

## Status
Proposed

## Date
2026-03-01

## Context / Problema
Current runtime baseline assumes split persistence:
- relational document/history storage (SQLite/SQLAlchemy),
- vector index storage (FAISS/numpy).

This creates architecture coupling if application code assumes this topology is mandatory.
Target requirement is technology-agnostic persistence that can support:
- split-store backends,
- unified engines (Elasticsearch, pgvector, Qdrant) via adapters.

Current state vs proposal:
- Implemented now: capability primitives (`StorageProfile`, `StorageCapability`) and `DURABLE_SAGA` orchestration for current split-store runtime.
- Not implemented yet: explicit `SplitStorePersistencePort` / `UnifiedKnowledgeStorePort` abstractions and unified-store adapter contract matrix.

## Decision (proposed)
1. Keep application/use-case logic topology-agnostic and capability-driven.
2. Introduce explicit persistence topology contracts in ports:
- `SplitStorePersistencePort` (document/vector/history seams),
- `UnifiedKnowledgeStorePort` (single backend abstraction exposing equivalent operations).
3. Select orchestration strategy by `StorageProfile` capabilities:
- `ATOMIC` when backend guarantees single-store atomicity,
- `DURABLE_SAGA` for split-store multi-step consistency.
4. Keep concrete backend selection and wiring inside `composition` and `infrastructure`.

## Consequences
Positive:
- Core/use-cases stop hardcoding SQL + local-vector-index assumptions.
- New backends can be integrated with limited blast radius.
- Cross-backend behavior can be validated with contract tests.

Negative:
- More adapter/wiring complexity.
- Contract test matrix maintenance cost increases.

## Safeguards / Tests (planned)
- [TODO] Add adapter contract tests that run the same behavior suite on:
  - split baseline (`SQLite + FAISS/numpy`),
  - unified candidates (`Elasticsearch`, `pgvector`, `Qdrant`).
- [TODO] Add architecture guardrails to block backend-name branching in `core/use_cases`.

## Open Questions
1. Minimal operation set required for unified engines in D1.1 (query, upsert, delete, history, diagnostics).
2. Which capabilities are mandatory vs optional in `StorageProfile`.
3. Whether index rebuild remains a first-class operation for unified engines or becomes no-op/delegated.

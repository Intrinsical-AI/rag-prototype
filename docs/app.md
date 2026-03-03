# Application Layers — Current Boundaries

La estructura del proyecto separa preocupaciones en capas concéntricas: `core/` (dominio y casos de uso), `infrastructure/` (adaptadores), `composition/` (DI), y dos transportes (`http/`, `cli_commands/`).

## Objetivo de diseño

- Casos de uso transport-agnostic en `core/use_cases/`.
- Composition root independiente del transporte: `composition/`.
- Escrituras centralizadas en `MutationCoordinator`.
- Routers/CLI finos, sin lógica de negocio distribuida.
- FastAPI como dependencia opcional (`[server]` extra).

---

## Submódulos clave

### `core/use_cases/`

Casos de uso y coordinación transport-agnostic:

- `docs_mutation.py`: `MutationCoordinator`, `MutationIntent`.
- `_batch_coordinator.py`: `MutationBatchCoordinator` (micro-batching y drenado acotado).
- `_mutation_saga_executor.py`: `MutationSagaExecutor` (saga durable + recovery).
- `docs_ingest.py`: ingesta de textos con salida a mutación canónica.
- `docs_import.py`: import de JSON (ChatGPT/Gemini) + delegación a ingesta.
- `rag_query.py`: ask_eval + history read path.
- `mutations.py`: wrapper compartido de ejecución de mutaciones API/CLI.
- `errors.py`: errores tipados de aplicación + `map_runtime_error`.
- `results.py`: DTOs de salida de casos de uso.

### `core/domain/`

- `profiles.py`: capabilities (`ATOMIC`, `DURABLE_SAGA`, `READ_ONLY`), `StorageProfileRegistry`.

### `core/ports/`

Contratos de puertos:

- `__init__.py`: `RetrieverPort`, `GeneratorPort`, `EmbedderPort`, etc.
- `contracts.py`: `DocsMutationPorts`, `IndexMutationPorts`, `MutationJournalPort`, etc.

### `composition/`

Composition root y lifecycle runtime (transport-neutral):

- `container.py` + `factory.py`: construcción de retriever/generator/rag runtime, cache/versionado de `RagService`, wiring de puertos para mutaciones, recovery de journal.
- `adapters.py`: builders de adaptadores de infraestructura.

### `http/routers/`

Adaptadores HTTP por bounded context:

- `docs.py`: `GET /docs`, `POST /docs`, `POST /docs/import`, `POST /docs/mutate`
- `rag_router.py`: `POST /ask`, `POST /ask_eval`, `GET /history`
- `index.py`: `POST /index/rebuild`
- `health.py`: `GET /health`, `GET /ready`, `GET /health/ollama`
- `openrouter.py`, `meta.py`

---

## Fronteras estrictas

- `core/{domain,ports,services}` no importan `infrastructure/`, `http/`, ni `composition/`.
- `core/use_cases/` no importa `http/` ni `fastapi`/`starlette`.
- `http/routers/*` no importan `infrastructure/*` directamente.
- `composition/` solo importa `http/` bajo `TYPE_CHECKING`.
- `cli_commands/` importa `core/` y `composition/`, nunca `http/`.

Estas reglas están cubiertas por tests de arquitectura.

---

## Write path canónico

### Entradas

- API: `POST /api/docs/mutate`
- API ingest/import: internamente transforman a `MutationIntent`
- CLI: `rag-mutate-docs`
- CLI ingest: internamente transforma a `MutationIntent`
- CLI bootstrap: `rag-bootstrap` usa `run_sample_data_ingestion` y delega en `MutationCoordinator`

### Ejecución

1. Construcción de `MutationIntent` (con `op_id` idempotente opcional).
2. `MutationCoordinator.execute(...)` (orquestador delgado).
3. `MutationBatchCoordinator` agrupa y drena (`max_batch_size`, `max_wait_ms`).
4. `MutationSagaExecutor` precomputa embeddings fuera de lock.
5. Lock multi-store + journal.
6. SQL commit.
7. Vector delta incremental (`apply_delta_atomic`).
8. Compensación/recovery si falla fase vector.
9. Invalidación de caché de `RagService`.

### Garantía

No se promete 2PC universal entre cualquier backend, pero sí garantía de operación `DURABLE_SAGA` por perfil de storage.

### Bootstrap (actual)

- `src/local_rag_backend/scripts/sample_data_ingestion.py` ya no usa una vía ETL paralela.
- La carga de `faq.csv` se materializa como `MutationIntent` (upsert + delete stale por prefijo).
- Eso preserva lock/journal/recovery y evita drift SQL/vector en bootstrap.

---

## Read path

`/ask` y `/ask_eval` usan `core/services/rag_runtime.py::RagService` con retriever/generator resueltos en `AppContainer`.

El offload de trabajo bloqueante está centralizado en `infrastructure/concurrency/blocking.py` con pools por tipo de tarea (`default|mutation|network|eval`).

Los locks de escritura/archivo están en `infrastructure/concurrency/locks/{file_lock.py,write_lock.py}`.

---

## Superficie v1.0 (breaking)

Se eliminaron endpoints/commands legacy de mutación:

- `/api/docs/upsert`
- `/api/docs/delete`
- `/api/docs/delete_by_external_id`
- `rag-upsert-docs`
- `rag-delete-docs`
- `rag-delete-external-ids`

La mutación write-enabled se hace solo por la superficie unificada (`/api/docs/mutate`, `rag-mutate-docs`).

## Artefactos de evaluación

El dataset de evaluación por defecto vive en `datasets/rag_eval_v1.jsonl` (raíz del repositorio), no dentro de `src/`. Puede sobreescribirse con `RAG_EVAL_DATASET_PATH`.

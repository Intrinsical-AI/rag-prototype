# App Layer (FastAPI) - Current Boundaries

La capa `app` es la capa de entrega y orquestación: traduce HTTP/CLI a casos de uso, aplica políticas operativas (locking, recovery, errores, observabilidad) y delega al core/infra a través de contratos.

## Objetivo de diseño

- Una sola capa de orquestación: `app/application`.
- Sin capa intermedia ambigua (`app/services` fue eliminada).
- Escrituras centralizadas en `MutationCoordinator`.
- Routers/CLI finos, sin lógica de negocio distribuida.

---

## Submódulos clave

### `app/application`

Casos de uso y coordinación transport-agnostic:

- `docs_mutation.py`: `MutationCoordinator`, `MutationIntent`.
- `docs_ingest_use_case.py`: ingesta de textos con salida a mutación canónica.
- `docs_import_use_case.py`: import de JSON (ChatGPT/Gemini) + delegación a ingesta.
- `docs_query_use_case.py`: consulta/listado de documentos.
- `rag_query_use_case.py`: ask_eval + history read path.
- `mutations.py`: wrapper compartido de ejecución de mutaciones API/CLI.
- `storage_profiles.py`: capabilities (`ATOMIC`, `DURABLE_SAGA`, `READ_ONLY`).

### `app/contracts`

Contratos de app-layer:

- `ports.py`: `DocsMutationPorts`, `IndexMutationPorts`, `MutationJournalPort`, etc.
- `results.py`: DTOs de salida de casos de uso.

### `app/wiring`

Builders por defecto de dependencias para mutación/index:

- `mutation_ports.py`

### `app/routers`

Adaptadores HTTP por bounded context:

- `docs.py`: `GET /docs`, `POST /docs`, `POST /docs/import`, `POST /docs/mutate`
- `rag_router.py`: `POST /ask`, `POST /ask_eval`, `GET /history`
- `index.py`: `POST /index/rebuild`
- `health.py`: `GET /health`, `GET /ready`, `GET /health/ollama`
- `openrouter.py`, `meta.py`

### `app/container.py` + `app/factory.py`

Composition root y lifecycle runtime:

- construcción de retriever/generator/rag runtime,
- cache/versionado de `RagService`,
- wiring de puertos para mutaciones,
- recovery de journal incompleto al startup/background.

---

## Fronteras estrictas

- `routers/*` no importan `infrastructure/*` directamente.
- `application/*` no importa `fastapi`, `routers` ni `schemas`.
- `core/*` no importa `app/*`.
- No se permiten imports a `app.services`.

Estas reglas están cubiertas por tests de arquitectura.

---

## Write path canónico

### Entradas

- API: `POST /api/docs/mutate`
- API ingest/import: internamente transforman a `MutationIntent`
- CLI: `rag-mutate-docs`
- CLI ingest: internamente transforma a `MutationIntent`

### Ejecución

1. Construcción de `MutationIntent` (con `op_id` idempotente opcional).
2. `MutationCoordinator.execute(...)`.
3. Lock multi-store + journal.
4. SQL commit.
5. Vector delta incremental (`apply_delta_atomic`).
6. Compensación/recovery si falla fase vector.
7. Invalidación de caché de `RagService`.

### Garantía

No se promete 2PC universal entre cualquier backend, pero sí garantía de operación `DURABLE_SAGA` por perfil de storage.

---

## Read path

`/ask` y `/ask_eval` usan `core/services/rag_runtime.py::RagService` con retriever/generator resueltos en `AppContainer`.

El offload de trabajo bloqueante está centralizado en `app/blocking.py` con pools por tipo de tarea (`default|mutation|network|eval`).

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

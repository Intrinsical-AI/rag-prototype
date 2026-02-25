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
- `docs_ingest.py`: ingesta de textos con salida a mutación canónica.
- `docs_import.py`: import de JSON (ChatGPT/Gemini) + delegación a ingesta.
- `docs_query.py`: consulta/listado de documentos.
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
- `wiring/mutation_ports.py`: builders por defecto de dependencias para mutación/index.

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

El offload de trabajo bloqueante está centralizado en `infrastructure/concurrency/blocking.py` con pools por tipo de tarea (`default|mutation|network|eval`).

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

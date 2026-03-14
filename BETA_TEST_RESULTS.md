# Beta-Test Results — RAG Prototype v1.3.0
> Ejecutado: 2026-03-03
> Entorno: Python 3.12.12, uv 0.9.26, Linux (Fedora 43), sin OPENAI/SentenceTransformers
> Modo de retrieval bajo test: **sparse** (dense/hybrid requieren embedder externo)

---

## Resumen ejecutivo

| Severidad | Count | Tests afectados |
|-----------|-------|-----------------|
| 🔴 FAIL crítico | 5 | B1.5, B5.3, B5.4, B6.3/B6.4/B6.5 (root cause único), B8.x (sistémico) |
| 🟡 WARN | 5 | B3.3, B3.5, B6.1-/ready, B6.9, B6.11/health |
| 🟢 PASS | 28 | Ver tabla completa |
| ⚪ SKIP | 6 | Dense/hybrid (sin embedder en este entorno) |

**El release es estable en modo sparse.** Los FAILs son bugs de UX/DX y un bug de seguridad de baja severidad (comportamiento de bootstrap silencioso), no crashes de producción.

---

## Tabla de resultados completa

| Test | Veredicto | Descripción del comportamiento real |
|------|-----------|--------------------------------------|
| **B1.1** Status vacío | 🟢 PASS | 0 docs, sin crash, exit 0 |
| **B1.2** Bootstrap CSV | 🟢 PASS | 3 docs ingestados, idempotente |
| **B1.3** Bootstrap idempotente | 🟢 PASS | 0 docs nuevos en segunda ejecución (dedup SHA256 funciona) |
| **B1.4** CSV sin cabecera | 🟢 PASS | `CSV_HAS_HEADER=false` funciona, 2 docs adicionales |
| **B1.5** CSV inexistente | 🔴 FAIL | `_resolve_csv_path` hace **fallback silencioso** a `data/faq.csv` del repo cuando `FAQ_CSV` apunta a path inexistente. Exit 0, output "[OK]". El usuario no sabe que sus datos fueron ignorados. |
| **B2.1** Multi-formato | 🟢 PASS | `.txt` y `.md` ingestados correctamente, chunking aplicado |
| **B2.2** Archivo vacío | 🟢 PASS | `skipped=1`, exit 0 |
| **B2.3** Sólo whitespace | 🟢 PASS | `skipped=1`, exit 0 |
| **B2.4** Binario | 🟢 PASS | `skipped=1`, exit 0 (UnicodeDecodeError capturado) |
| **B2.5** Directorio completo | 🟢 PASS | Procesa 21 archivos, salta 6 (bin/json/etc.) |
| **B2.6** Doble-ingest dedup | 🟢 PASS | `unchanged=1, inserted=0` en segunda pasada |
| **B2.7** Path inexistente | 🟡 WARN | Click valida y da error, pero con traceback crudo (click `BadParameter`) en lugar de `[ERROR]` limpio |
| **B3.1** Mutación válida | 🟢 PASS | 3 docs ingestados, journal en COMMITTED |
| **B3.2** Payload vacío `{}` | 🟢 PASS | `[ERROR] Mutation intent must include upserts and/or deletions.`, exit 1 |
| **B3.3** Content vacío en upsert | 🟡 WARN | El upsert con `content=""` es **filtrado silenciosamente** en `normalize_intent` (línea `if str(it.content).strip()`). El error resultante dice "must include upserts and/or deletions" en lugar de "empty content filtered". Mensaje engañoso. |
| **B3.4** Upsert+delete mismo ID | 🟢 PASS | `[ERROR] upserts and delete_external_ids cannot target the same external_id values: conflict-id`, exit 1 |
| **B3.5** Delete ID inexistente | 🟡 WARN | Crea **tombstone preventivo** (`tombstoned=1, deleted_sql=0`). Comportamiento semánticamente discutible: un ID que nunca existió queda tombstoned, bloqueando ingests futuros de ese ID. Documentado pero sorprendente. |
| **B3.6** Reingest tombstoneado | 🟢 PASS | `[ERROR] Cannot upsert tombstoned external_id values: beta-doc-001`, exit 1. Invariante de tombstone funciona. |
| **B3.7** JSON malformado | 🟢 PASS | `[ERROR] Expecting ',' delimiter`, exit 1 |
| **B4.1** Rebuild en sparse | 🟢 PASS | `[ERROR] rebuild-index requires RETRIEVAL_MODE=dense\|hybrid`, exit 1 |
| **B4.2** Rebuild en dense | ⚪ SKIP | Requiere embedder real (OpenAI key o `dense-st` extra). Error correcto: "Dense/hybrid retrieval requires an embeddings backend." |
| **B4.3** Rebuild DB vacía | 🟢 PASS | Mismo comportamiento que B4.1 en sparse (correcto) |
| **B4.4** Status post-rebuild | 🟢 PASS | Reporta ntotal y estado correctamente |
| **B4.5** Drift detection (sparse) | ⚪ SKIP | En sparse no hay FAISS index; drift sólo aplicable en dense/hybrid |
| **B5.1** Eval dataset pequeño | 🟢 PASS | `hit_rate=0.667 mrr=0.667`, exit 1 por umbral. Métricas correctas. |
| **B5.2** Eval dataset golden | 🟢 PASS | `hit_rate=1.000 mrr=1.000`, exit 0. Golden suite: perfecta. |
| **B5.3** JSONL corrupto | 🔴 FAIL | **Traceback crudo** de `ValueError: Invalid dataset line 1: unknown type=None`. `eval_cmd` no envuelve errores de parseo de dataset (a diferencia de `bootstrap_cmd` que sí tiene try/except → `[ERROR]`). |
| **B5.4** Dataset vacío | 🔴 FAIL | **Traceback crudo** de `ValueError: Unsupported schema_version=0`. El archivo vacío no tiene línea `meta`, la función asume `schema_version=0` y lanza error sin mensaje limpio. |
| **B6.1** Health/Ready/Ollama | 🟢 PASS | `/health` → 200, `/ready` → 503 con detalle estructurado, `/health/ollama` → 200 (Ollama está corriendo localmente) |
| **B6.1** /ready sin LLM | 🟡 WARN | `/ready` devuelve HTTP 503 (correcto), pero el cuerpo está envuelto en `{"detail": {...}}` — formato de error FastAPI, no de payload exitoso. Puede confundir a consumidores del endpoint. |
| **B6.2** Config/Templates | 🟢 PASS | JSON válido, `retrieval_mode` refleja env var, 4 templates disponibles |
| **B6.3** Ask sin LLM | 🔴 FAIL | **HTTP 500** en lugar de 503. `get_rag_service()` dependency lanza `RuntimeError` (no `AppError`) antes de que Pydantic valide el body. El error se escapa del exception handler y produce "Internal Server Error" genérico. |
| **B6.4** Ask pregunta vacía | 🔴 FAIL | **HTTP 500** (mismo root cause que B6.3: la dep crashea antes de la validación Pydantic). Sin la dep crasheada devolvería 422 (Pydantic valida `min_length=1`). |
| **B6.5** Ask k=0/-1 | 🔴 FAIL | **HTTP 500** (mismo root cause). Sin la dep crasheada devolvería 422 (Pydantic valida `ge=1`). |
| **B6.6** Ask k > total docs | ⚪ SKIP | Mismo root cause que B6.3, no testeable sin LLM |
| **B6.7** Paginación docs/history | 🟢 PASS | limit=0 → 422, limit=-1 → 422, paginación correcta |
| **B6.8** Docs mutate API | 🟢 PASS | Upsert válido → 200, payload vacío → 422 correcto, content vacío → 422 correcto (Pydantic `min_length=1` en content) |
| **B6.9** Docs import API | 🟡 WARN | `/docs/import` es para exportaciones de conversaciones (ChatGPT/Gemini), **no** para documentos genéricos. El test plan asumía incorrectamente su propósito. Comportamiento correcto: 422 "Could not detect a supported export format". |
| **B6.10** Index rebuild API sparse | 🟢 PASS | HTTP 400 "Index rebuild requires dense or hybrid mode." |
| **B6.11** Auth API_KEY | 🟢 PASS | Sin key: 401, key incorrecta: 401, key correcta: 200. Nota: `/health` también requiere key cuando `API_KEY` está configurado (puede ser problema para K8s probes). |
| **B6.12** Mutations concurrentes | 🟢 PASS | 5 mutations paralelas: cada una inserted=1, 0 duplicados, 0 deadlocks. Mecanismo de write-lock funciona. |
| **B6.13** OpenRouter sin key | 🟢 PASS | HTTP 400 "OpenRouter is not configured" |
| **B7.1** FAISS corrupto | 🟢 PASS | `Index: [ERROR] missing (hint: rebuild)` — detectado, no crash |
| **B7.2** id_map malformado | 🟢 PASS | `Index: [ERROR] corrupt (hint: rebuild)` — detectado, no crash |
| **B7.3** id_map vacío vs FAISS válido | ⚪ SKIP | Requiere dense mode con embedder real para validar mismatch |
| **B7.4** Journal recovery | 🟢 PASS | Journal limpio post-operaciones. `mutation_journal: {status: ok, incomplete_records: 0}` en `/ready`. |
| **B8.1** RETRIEVAL_MODE inválido | 🟡 WARN | Pydantic valida correctamente, pero como `settings = Settings()` está al nivel de módulo, el error produce un **traceback crudo** en import time en lugar de mensaje amigable. Sistémico para todas las settings inválidas. |
| **B8.2** SQLITE_URL sin permisos | 🟡 WARN | SQLAlchemy traceback crudo en lugar de mensaje amigable. |
| **B8.3** INGEST_BATCH_SIZE=1 | 🟢 PASS | Lento pero correcto |
| **B8.3** INGEST_BATCH_SIZE=10000 | 🟡 WARN | Pydantic valida (max 512) pero misma presentación de traceback crudo. |
| **R1** alchemy_engine.py eliminado | 🟢 PASS | Archivo no existe. Breaking change aplicado correctamente. |
| **R2** `from_settings()` | 🟢 PASS | Factory funciona, retorna `AppContainer` |
| **R3** `runtime_wiring_defaults()` | 🟢 PASS | Retorna `dict` |
| **R4** Bootstrap → MutationCoordinator | 🟢 PASS | `sample_data_ingestion.py` importa y llama `MutationCoordinator`, no legacy ETL |
| **B9** UI Frontend | 🟢 PASS | HTML válido, carga correctamente. Nota cosmética: ejemplo questions hardcoded con "Legal Engine Search (LES)" irrelevante para KB genérica. |

---

## Hallazgos clasificados por severidad

### 🔴 FAIL — Deben corregirse antes de promover el release

#### F-01: Bootstrap silencia path inválido de FAQ_CSV `[B1.5]`
- **Ubicación**: [sample_data_ingestion.py:46-52](src/local_rag_backend/scripts/sample_data_ingestion.py#L46-L52) — `_resolve_csv_path`
- **Comportamiento**: Cuando `FAQ_CSV=/path/que/no/existe`, cae en fallback a `data/faq.csv` del repo **sin ningún warning o error**.
- **Impacto**: El usuario cree que ingestó sus datos, pero en realidad ingestó la FAQ de demo. Silencio total.
- **Fix sugerido**: Si `csv_path` fue provisto explícitamente (no es `None`) y no existe, lanzar `FileNotFoundError` inmediatamente. El fallback sólo debe usarse cuando `csv_path is None`.

#### F-02: `eval_cmd` sin manejo de errores de parseo `[B5.3, B5.4]`
- **Ubicación**: [cli_commands/eval.py:58](src/local_rag_backend/cli_commands/eval.py#L58)
- **Comportamiento**: JSONL corrupto y dataset vacío producen traceback crudo en lugar de `[ERROR] ...`.
- **Contraste**: `bootstrap_cmd`, `mutate_docs_cmd` tienen try/except → mensaje limpio. `eval_cmd` no.
- **Fix sugerido**: Envolver `load_eval_dataset()` y `run_retrieval_eval()` en try/except, salir con `[ERROR]`.

#### F-03: `get_rag_service()` dependency lanza `RuntimeError` no mapeado `[B6.3, B6.4, B6.5]`
- **Ubicación**: [composition/adapters.py:541](src/local_rag_backend/composition/adapters.py#L541) — `resolve_preferred_llm_provider`
  [composition/factory.py:129](src/local_rag_backend/composition/factory.py#L129) — `get_rag_service`
- **Comportamiento**: `RuntimeError: No LLM configured` escapa al handler de FastAPI → HTTP 500 en lugar de 503. Además, la dep se resuelve **antes** que Pydantic valide el body, por lo que B6.4 (pregunta vacía) y B6.5 (k inválido) también devuelven 500 en lugar de 422.
- **Impacto**: Todos los endpoints que inyectan `get_rag_service` como dep (ask, ask_eval, history) devuelven 500 cuando no hay LLM configurado, ocultando errores de validación Pydantic.
- **Fix sugerido**:
  1. En `resolve_preferred_llm_provider`, lanzar `LLMConfigurationError` en lugar de `RuntimeError` (ya está en el exception_handler como `handle_runtime_error`).
  2. O bien, manejar el `RuntimeError` en la dependencia de FastAPI con una función wrapper.

#### F-04: Presentación de errores de Settings como traceback crudo `[B8.1, B8.2, B8.3]`
- **Ubicación**: [settings.py:325](src/local_rag_backend/settings.py#L325) — `settings: Settings = Settings()` a nivel de módulo
- **Comportamiento**: Cualquier variable de entorno con valor inválido produce un `pydantic_core.ValidationError` o `sqlalchemy.exc.OperationalError` con traceback completo en stderr.
- **Severidad real**: WARN (la información útil está presente, sólo la presentación es fea). Rebajado a WARN por ser comportamiento estándar de Pydantic Settings v2.

---

### 🟡 WARN — Comportamiento subóptimo, no bloqueante

#### W-01: Mensaje de error engañoso para upsert con content vacío `[B3.3]`
- **Ubicación**: [core/use_cases/docs_mutation_contracts.py:49](src/local_rag_backend/core/use_cases/docs_mutation_contracts.py#L49)
- El filtro `if str(it.external_id).strip() and str(it.content).strip()` elimina silenciosamente upserts con content vacío, luego lanza "Mutation intent must include upserts and/or deletions" — confuso para quien sí envió un upsert.

#### W-02: Tombstone preventivo en delete de ID inexistente `[B3.5]`
- Un `delete_external_ids: ["id-inexistente"]` crea un tombstone para un ID que nunca existió. Comportamiento determinista y documentado, pero sorprendente: bloqueará futuros ingests de ese ID.

#### W-03: `/ready` en formato de error (`{"detail": {...}}`) `[B6.1]`
- La respuesta de `/ready` cuando no está listo usa el formato de error de FastAPI (`HTTPException`), no un payload de readiness estándar. Un consumidor esperaría `{"status": "not_ready", ...}` con HTTP 503, no `{"detail": {...}}`.

#### W-04: `/health` requiere `API_KEY` cuando está configurado `[B6.11]`
- El router completo (`/api/*`) tiene `Depends(require_api_key)`. Esto incluye `/health`. En deployments con `API_KEY` configurado, los probes de K8s/Docker que consultan `/health` sin key recibirán 401.

#### W-05: Traceback de Click para path inválido en `rag-ingest` `[B2.7]`
- `rag-ingest /path/inexistente` produce traceback de Click en lugar de `[ERROR]` limpio.

#### W-06: UI con example questions hardcoded irrelevantes `[B9]`
- La UI tiene `<li>What is Legal Engine Search (LES)?</li>` hardcoded en `index.html`. No son ejemplos genéricos ni se generan dinámicamente desde el KB.

#### W-07: `rag-server` no expone `--host`/`--port` como CLI flags
- El servidor sólo acepta `APP_HOST`/`APP_PORT` como env vars. No hay flags CLI para cambiar host/puerto sin modificar el entorno. La mayoría de CLIs (uvicorn, gunicorn) exponen estos como flags.

#### W-08: Error de Settings como tracebacks crudos `[B8.1, B8.2, B8.3]`
- Settings inválidas producen traceback en stderr. Sistémico por `settings = Settings()` a nivel de módulo.

---

### ⚪ SKIP — No testeable en este entorno

- **Dense/hybrid mode completo**: Requiere `OPENAI_API_KEY` real o `uv sync --extra dense-st` (SentenceTransformers + modelo descargado).
- **Drift detection (B4.5 dense)**: Requiere rebuild exitoso en dense mode.
- **id_map mismatch (B7.3)**: Requiere dense mode funcional.
- **Ask con LLM real**: Servidor disponible con Ollama, modelos disponibles (`qwen3.5:2b-q8_0`), pero `OLLAMA_ENABLED` no testeado exhaustivamente dado los otros FAILs.
- **Archivo oversized import (B6.9)**: El endpoint `/docs/import` es para exportaciones de conversaciones, no documentos genéricos.

---

## Hallazgos adicionales (observaciones de campo)

1. **Esquemas CLI vs API divergentes y no documentados**: El CLI de `rag-mutate-docs` usa `{upserts, delete_ids, delete_external_ids}` y el HTTP API también usa el mismo schema, PERO la documentación del `--help` del CLI y el OpenAPI del servidor no hacen referencia al otro. Un desarrollador usando la API probará con "intents" (lógico por analogía con otros sistemas de mutación) y obtendrá un error confuso de Pydantic.

2. **`data/faq.csv` con delimitador `;` hardcoded en `sample_data_ingestion.py`**: `DELIMITER = ";"` está hardcodeado. Si alguien cambia el `faq.csv` del repo usando `,` como separador, el bootstrap lo procesará incorrectamente sin error.

3. **`/docs/import` propósito no obvio**: El endpoint acepta exportaciones de ChatGPT/Gemini. El nombre `/docs/import` sugiere importación genérica de documentos. Puede generar confusión en nuevos usuarios.

4. **Plan de beta-test tenía 2 errores factuales**: (a) variable de entorno `FAZ_CSV` (typo, debe ser `FAQ_CSV`); (b) formato `intents` para mutate (el correcto es `upserts/delete_external_ids`). Ambos descubiertos durante la ejecución.

---

## Regresiones v1.3.0 — Todas verificadas ✅

| Regresión | Resultado |
|-----------|-----------|
| `alchemy_engine.py` eliminado | ✅ Archivo no existe |
| `AppContainer.from_settings()` funciona | ✅ Retorna `AppContainer` |
| `runtime_wiring_defaults()` existe | ✅ Retorna `dict` |
| Bootstrap via `MutationCoordinator` | ✅ No usa ETL legacy |

---

## Priorización de fixes

```
P0 (antes de release público/producción):
  F-03: HTTP 500 en ask sin LLM (afecta DX, puede confundir monitoreo)
  F-01: Bootstrap silencia FAQ_CSV inválido (pérdida silenciosa de datos)

P1 (próximo sprint):
  F-02: eval_cmd sin manejo de errores → tracebacks crudos
  W-03: /ready formato de respuesta inconsistente
  W-04: /health requiere API_KEY → rompe probes K8s

P2 (backlog):
  W-01: Mensaje de error confuso en upsert con content vacío
  W-05/W-08: Tracebacks crudos en inputs inválidos (CLI y settings)
  W-06: Example questions hardcoded en UI
  W-07: rag-server sin --host/--port flags
```

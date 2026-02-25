# App 

App (FastAPI) (/src/local_rag_backend/app)
La capa app es la capa de entrega/orquestación: convierte HTTP en casos de uso, aplica seguridad, observabilidad y concurrencia, y estandariza errores sin contaminar el dominio. No sustituye a core ni a infrastructure; los compone (instancia + conecta + ejecuta).

El punto de entrada es main.py:64 donde se crea el FastAPI(lifespan=lifespan). El router raíz agrega los seis subrouters en api_router.py:17-23: health, rag, docs, index, openrouter, meta.

Además de delivery HTTP, main.py tiene dos responsabilidades adicionales: sirve un SPA frontend en / y /assets/{path} con lógica dual (resource de paquete instalado o directorio de repo, main.py:96-146), y gestiona CORS (main.py:68-76) — permisivo (*) en debug mode, allow_credentials=False forzado (combinación * + credentials es inválida por spec CORS).

Blocking
La ejecución de trabajo síncrono desde handlers async se centraliza en run_blocking (blocking.py:141), con:

Pools por tipo de tarea: default|mutation|network|eval (blocking.py:27).
Workers y límite de cola por tipo con override por env (RAG_BLOCKING_WORKERS_<TYPE>, RAG_BLOCKING_QUEUE_<TYPE>) (blocking.py:65-79).
La capacidad efectiva de "pending" es max(1, workers + queue_limit) (blocking.py:128) — no solo el queue_limit. Para default (8 workers + 64 cola) = 72 slots totales.
Backpressure por rechazo inmediato: si pending >= max_pending, se lanza RuntimeError (blocking.py:153-162) — el modelo es reject, no slow-down. Este error no tiene mapping explícito a 429 en error_mapping.py.
El mecanismo de ejecución es polling (await asyncio.sleep(0.001) mientras not fut.done(), blocking.py:189-190) en lugar de loop.run_in_executor(). La razón está documentada en el código (blocking.py:169-172): evitar deadlocks bajo ciertos ASGI test harnesses. El tradeoff es hasta ~1 ms de jitter adicional por tarea.
Instrumentación de tiempo de espera en cola (blocking.py:179-182), saturación y duración vía telemetry (blocking.py:201-205).
Telemetry
TelemetrySink define el contrato abstracto via Protocol (telemetry.py:20), con seis métodos: log_event, observe_query, observe_ingest, observe_blocking_queue, observe_blocking_queue_wait, observe_blocking_run.

PrometheusTelemetry implementa:

Métricas de query: total y latencia (telemetry.py:59-68).
Métricas de ingesta: requests y documentos (telemetry.py:69-78).
Métricas de salud del offload: pending, capacity, saturation ratio, wait, duración (telemetry.py:81-110).
Logging estructurado JSON: json.dumps sobre stdlib logging (no structlog) (telemetry.py:112-117).
Sink swappable con get_telemetry/set_telemetry/reset_telemetry (telemetry.py:180-191).
El módulo observability.py actúa como fachada delgada sobre telemetry: expone log_event, observe_query, observe_ingest, Timer y fingerprint_question — que hashea la pregunta en SHA-256 truncado para logging sin exponer contenido del usuario (observability.py:55-57).

Monitoring
MetricsMiddleware emite http_requests_total y latencias por método/path (middleware.py:37). Solo se añade al pipeline si settings.enable_monitoring es true (main.py:78-79).
Etiquetado de paths de baja cardinalidad para evitar explosión de series: usa request.scope["route"].path si hay match, <unmatched> para 404, y /assets/* para assets estáticos (middleware.py:52-72).
El endpoint /metrics está siempre registrado independientemente de enable_monitoring (main.py:84-93); si monitoring está deshabilitado, get_metrics() devuelve un comentario en texto plano.
Fallback no-op completo si prometheus_client no está instalado: _NoopMetric implementa labels/inc/observe/set como no-ops (metrics_backend.py:15-45), con la importación condicional en metrics_backend.py:48-69.
Security
Seguridad mínima pero explícita, con dos capas:

Startup: enforce_safe_bind_config() (security.py:84) se invoca en el lifespan (main.py:46). Lanza RuntimeError si el host de bind no es localhost y no hay API key configurada.

Per-request: require_api_key (security.py:108) tiene tres paths:

Si hay API key configurada: exige X-API-Key via hmac.compare_digest (security.py:128-130).
Si no hay API key y public_bind_requires_api_key=False: pasa libremente (security.py:116-117).
Si no hay API key pero el flag es true (default): verifica que todos los hosts resueltos sean localhost; cualquier host desconocido es rechazado con 401 (security.py:118-125).
Validación de origen con X-Forwarded-For (security.py:42-50) y Forwarded RFC 7239 (security.py:52-60) con enfoque fail-closed: cadena proxy presente pero no parseable → token __proxy_ambiguous__ que nunca pasa la validación.

Protege /api (main.py:81) y /metrics (main.py:88).

Diagnostics
Diagnóstico operativo en diagnostics.py:

SQL counts: get_documents_count y get_history_count via SQLAlchemy Engine (diagnostics.py:16-23).
get_retrieval_index_stats (diagnostics.py:151): devuelve status: ok|missing|corrupt|drift con hints de remediación.
Integridad del id_map (mismatch vectores/id_map, duplicados) en _apply_vector_integrity_status (diagnostics.py:81).
Validación del manifest e inferencia de drift en _apply_manifest_status (diagnostics.py:96).
Readiness compone DB + SQL counts + RAG service + LLM providers + índice en health.py:63-90.
Dependencies
La DI HTTP es delgada y estable en dependencies.py:

get_app_context() (:21): shim síncrono hacia factory._get_app_context().
get_app_container_dependency() (:25): shim async — evita el threadpool offload de FastAPI para una lectura que es O(1).
get_settings_dependency() (:30): ídem para settings.
Los nombres _dependency son intencionales: señalan que son Depends-callables de FastAPI, no utilidades directas.

Container
AppContainer (container.py:70) es el composition root de la capa app. Recibe más de 20 parámetros keyword-only en __init__ (container.py:75-104) — todos son factories inyectables principalmente para testabilidad (ver _collect_container_overrides en factory.py).

Responsabilidades:

Construye retriever por configuración de request (container.py:203), generator (container.py:230) y RagService default (container.py:245).
Expone puertos de mutación: docs_mutation_ports() (container.py:169) e index_mutation_ports() (container.py:188).
Centraliza locking vía run_multi_store_write_locked() (container.py:165).
Cachea RagService con invalidación por versión: get_rag_service() (container.py:282) lee la versión de SystemStateStorage (SQLite), compara con la versión cacheada localmente, y reconstruye solo si hay cambio. reset_rag_service() (container.py:297) hace bump en SQLite + limpia la caché local.
App Context y Factory
AppContext es un dataclass frozen+slots que empaqueta settings + container como contexto runtime (app_context.py:13).

factory.py gestiona el singleton de AppContext con dos niveles de caching:

Singleton de AppContext (_APP_CONTEXT, factory.py:48-49) con DCL (double-checked locking) en get_app_context() (factory.py:126).
Caché de RagService dentro del container (ver §Container anterior).
reset_rag_service() (factory.py:156) opera en ambos niveles: (1) llama container.reset_rag_service() para bump SQLite + clear caché local del container, y (2) llama reset_app_context() para anular el singleton. Edge case: si _APP_CONTEXT is None al momento del reset, crea un container temporal solo para hacer el bump en SQLite (factory.py:159).

_collect_container_overrides() (factory.py:52) detecta monkeypatching de tests comparando referencias de módulo, y propaga solo los overrides explícitos al AppContainer.

Services (subpackage)
app/services/ es la capa de orquestación sincrónica entre application/ y core, con cinco módulos:

docs.py: lógica de mutación consolidada (ingest, upsert, delete by id, delete by external_id) que application/ y los routers invocan (services/docs.py).
index.py: lógica de rebuild de índice.
openrouter.py: lógica del proxy OpenRouter.
mutation_ports.py: construcción de DocsMutationPorts e IndexMutationPorts — agrupaciones de adapters que las operaciones de mutación consumen.
ports.py: definición de los tipos DocsMutationPorts e IndexMutationPorts.
La relación jerárquica es: routers/ → application/ → services/ → core.

Application
Casos de uso y orquestación en app/application/:

docs.py: parsing/validación de imports (ChatGPT/Gemini), list_docs_page_sync, execute_import_docs_sync (application/docs.py:70).
health.py: checks de DB, SQL counts, índice e integridad para readiness.
index.py: rebuild_index_sync.
mutations.py: pipeline común API/CLI con dos variantes:
run_api_mutation (async): lock + offload + error mapping + reset (application/mutations.py:34).
run_cli_mutation (sync): ensure_schema + lock + error mapping + reset (application/mutations.py:58).
rag.py: execute_ask_eval_sync y list_history_entries_sync.
evaluation.py: eval offline.
openrouter.py: lógica del proxy OpenRouter.
Routers
/docs (routers/docs.py): list, ingest, upsert, delete, delete_by_external_id, import. Cada mutación sigue el patrón run_api_mutation(operation, run_locked, reset_after, map_error) (docs.py:129). Error mapping específico por endpoint: ingest → BadRequest, upsert → Conflict/BadRequest, import → 413/422/400.
/health (routers/health.py): liveness GET /health (health.py:53), readiness GET /ready (health.py:63), y GET /health/ollama que hace HTTP GET al servidor Ollama vía pool network (health.py:93).
/index (routers/index.py): rebuild bajo lock con validación de modo (dense o hybrid requerido, index.py:34).
/openrouter (routers/openrouter.py): proxy en pool network (openrouter.py:58-62); verifica openrouter_enabled y openrouter_api_key antes de despachar (openrouter.py:42-48). Contiene un assert de producción en :65 (deuda técnica menor).
/rag (routers/rag.py): POST /ask con RagService cacheado (rag.py:86); GET /history (rag.py:128); POST /ask_eval con config efímera por request en pool eval (rag.py:153), con validación previa via container.validate_rag_config().
/meta (routers/meta.py): plantillas hardcodeadas en cuatro variantes (meta.py:23) y config disponible (meta.py:57) (retrieval_mode, hybrid_alpha, temperature, max_tokens, available_providers).
Middleware
MetricsMiddleware (middleware.py:37) es la capa transversal de observabilidad HTTP. Se añade condicionalmente en main.py:78-79. Re-exporta los símbolos de metrics_backend a nivel de módulo (middleware.py:23-28) para que los tests puedan monkeypatchear el middleware directamente sin tocar el backend.
# Roadmap 
## Brainstorm
- Ingestion Pipeline - ¿Flex. base skeleton + adapters?
    -- Extract
    --- Suport formats: .csv, .txt, .md, .pdf [F]
    --- metadata contract
    -- Clean
    -- Chunk
        --- Fixed (logs, flat texts) chunk size and overlap
        --- Parent-Doc Retrieval [F]
        --- Custom heuristics RegEx based (emails structured docs) [F]
        ** Caution with tables; easy to break semantic coherence intrachunk
    -- Dedup (hash-based)
        --- Add external_id/source_id + content_hash + version
    -- Embed 
        --- Re-embedding policy / updating
    -- Store
    -- Update (Insert/Upsert) policy

- Retrieval 
    -- Bi-Encoders
    -- Cross-Encoders [F]
    -- ColBERT / ColBERTv2 [F]

- Monitoring Module
    -- Opentelemetry / Prometheus

- Evaluation Module
    -- Offline Metrics
    -- Online Metrics

- Safety
    -- PII [F]
    -- Direct Prompt Injection
    -- Indirect Prompt Injection 


## Proposal
Release 02-2026 (objetivo: 10 PRs, stacked):

1. **PR1: Operabilidad y diagnóstico (status/ready)** [DONE]
- Scope: mejorar `rag-status` y `/api/ready` para reportar consistencia y causas.
- DoD: `rag-status` incluye conteos (docs/vectores) y paths; `/api/ready` da error 503 con detalle accionable si falta índice en `dense/hybrid` o hay drift/corrupcion.

2. **PR2: Identidad de documento (external_id) + modelo de metadata** [DONE]
- Scope: añadir `external_id/source_id`, `metadata`, `content_hash`, timestamps; contrato estable.
- DoD: schema + migración; tests de persistencia y lectura; docs actualizadas de identidad.

3. **PR3: Upsert idempotente (API/CLI)** [DONE]
- Scope: endpoint/CLI `upsert` por `external_id` y política de actualización.
- DoD: re-ingesta misma fuente no duplica; update de contenido actualiza solo lo afectado (o marca rebuild requerido); tests de idempotencia/update.

4. **PR4: Estabilización: estructura de paquete y límites de capas (refactor sin cambios funcionales)**
- Scope:
  - Reducir deuda de organización: evitar "cajón de sastre" en `src/local_rag_backend/*` raíz.
  - Mover piezas a su capa natural (sin romper compatibilidad de imports).
- Propuesta concreta (migración completa, sin shims):
  - Eliminar módulos ambiguos en raíz y actualizar imports a ubicaciones definitivas:
    - API schemas (Pydantic) -> `src/local_rag_backend/app/schemas.py`
    - Prompting (lógica core) -> `src/local_rag_backend/core/services/prompting.py`
    - Diagnostics (ready/status) -> `src/local_rag_backend/app/diagnostics.py`
    - Text processing / corpus helpers -> `src/local_rag_backend/core/services/text_processing.py` y `src/local_rag_backend/core/services/corpus.py`
- DoD:
  - Cero cambios funcionales (solo movimiento/organización).
  - Superficie reducida: no quedan `models.py/prompting.py/diagnostics.py/utils.py` en raíz.
  - Tests + mypy + ruff + black en verde.
  - Docs actualizadas (rutas nuevas como source of truth).

5. **PR5: Extract adapters (txt/md/csv) + `rag-ingest` por path/dir**
- Scope: loaders por formato + CLI para ingestar ficheros/directorios.
- DoD: ingest de dir mixto; límites/tamaños; tests con fixtures.
- Quick-wins:
  - Factory/Strategy para loaders (p.ej. `infrastructure/ingestion/loaders/factory.py`).
  - No fiarse solo de extensión: detección best-effort por bytes/heurística y `python-magic` como extra opcional.

6. **PR6: Clean + Chunk pipeline configurable**
- Scope: normalización/cleaning y chunker determinista configurable por settings.
- DoD: chunking determinista; tests de boundaries/overlap/metadata.
- Nota: preparar metadata para features futuras (chunk_index, parent_doc_id) sin introducir aún Parent-Doc Retrieval.

7. **PR7: Dedup hash-based + constraints**
- Scope: hashing por chunk + constraint/índice SQL para dedup.
- DoD: reingesta => 0 inserts; cambio de `chunker_version` => re-chunk esperado; tests de dedup.
- Matiz: el hash debe incluir versión de chunking y modelo de embeddings para evitar falsos "ya existe":
  - `sha256(cleaned_text + chunker_version + embedding_model_name)`

8. **PR8: Borrado por `external_id` + (opcional) tombstones**
- Scope: delete consistente por identidad; decidir hard vs soft delete.
- DoD: delete no reaparece tras rebuild; dense/hybrid consistente; tests delete+ask+rebuild.

9. **PR9: Manifest del índice + detección de drift (model/dim/chunker)**
- Scope: `index_manifest.json` y validación en `/ready`/CLI.
- DoD: mismatch detectado y explicado; rebuild corrige; tests de mismatch.
- Quick-win: `meta.json`/`manifest.json` junto a `index.faiss` con `{embedding_model, dimension, chunker, created_at}`.

10. **PR10: Evaluación + monitoring mínimo + retrieval quality (reranker opcional)**
- Scope:
  - Evaluación offline como gate (`rag-eval`) (dataset versionado + métricas).
  - Monitoring mínimo (métricas y logs estructurados ingest/query).
  - Reranker opcional detrás de flag (solo si hay dataset para medir mejora).
- DoD:
  - `rag-eval` reproducible (subset rápido en CI) y falla por regresión.
  - `/metrics` consistente cuando habilitado + smoke tests.
  - Reranker toggle seguro + tests.

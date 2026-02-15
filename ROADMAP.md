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
1. **PR1: Operabilidad y diagnóstico (status/ready)**
- Scope: mejorar `rag-status` y `/api/ready` para reportar consistencia y causas.
- DoD: `rag-status` incluye conteos (docs/vectores) y paths; `/api/ready` da error 503 con detalle accionable si falta índice en `dense/hybrid`.

2. **PR2: Identidad de documento (external_id) + modelo de metadata**
- Scope: añadir `external_id/source_id`, `metadata`, `content_hash`, timestamps; contrato estable.
- DoD: schema + migración; tests de persistencia y lectura; docs actualizadas de identidad.

3. **PR3: Upsert idempotente (API/CLI)**
- Scope: endpoint/CLI `upsert` por `external_id` y política de actualización.
- DoD: re-ingesta misma fuente no duplica; update de contenido actualiza solo lo afectado (o marca rebuild requerido); tests de idempotencia/update.

4. **PR4: Extract adapters (txt/md/csv) + `rag-ingest` por path/dir**
- Scope: loaders por formato + CLI para ingestar ficheros/directorios.
- DoD: ingest de dir mixto; límites/tamaños; tests con fixtures.

5. **PR5: Clean + Chunk pipeline configurable**
- Scope: normalización/cleaning y chunker determinista configurable por settings.
- DoD: chunking determinista; tests de boundaries/overlap/metadata.

6. **PR6: Dedup hash-based + constraints**
- Scope: hashing por chunk + constraint/índice SQL para dedup.
- DoD: reingesta => 0 inserts; cambio de `chunker_version` => re-chunk esperado; tests de dedup.

7. **PR7: Borrado por `external_id` + (opcional) tombstones**
- Scope: delete consistente por identidad; decidir hard vs soft delete.
- DoD: delete no reaparece tras rebuild; dense/hybrid consistente; tests delete+ask+rebuild.

8. **PR8: Manifest del índice + detección de drift (model/dim/chunker)**
- Scope: `index_manifest.json` y validación en `/ready`/CLI.
- DoD: mismatch detectado y explicado; rebuild corrige; tests de mismatch.

9. **PR9: Evaluación offline como gate (`rag-eval`)**
- Scope: dataset versionado + script que compute recall@k/MRR@k/latencia.
- DoD: comando reproducible; subset rápido en CI; tests de parser/reporte.

10. **PR10: Retrieval quality (reranker opcional) + monitoring mínimo**
- Scope: reranker opcional (flag) y métricas/logs estructurados ingest/query.
- DoD: toggle seguro; latencias medidas; `/metrics` consistente cuando habilitado; tests smoke de métricas y reranker.

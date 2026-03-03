# Beta-Test Plan — RAG Prototype v1.3.0
> Aliado hostil: el objetivo es romper el sistema, no confirmarlo.
> Ejecutar en orden: cada sección asume el estado del anterior salvo que se indique reset.

---

## Preparación del entorno

```bash
# Directorio de trabajo temporal con estado limpio
export TEST_DIR=$(mktemp -d /tmp/rag-beta-XXXXXX)
export SQLITE_URL="sqlite:///$TEST_DIR/test.db"
export INDEX_PATH="$TEST_DIR/faiss.index"
export ID_MAP_PATH="$TEST_DIR/id_map.json"
export DATA_DIR="$TEST_DIR/data"

mkdir -p "$TEST_DIR"

# Instalar en entorno limpio
cd /path/to/rag-prototype
uv sync --extra server

# Alias para no repetir vars
alias rag-cli="uv run --no-sync"
```

### Datasets sintéticos

```bash
# FAQ mínima (3 entradas)
cat > "$TEST_DIR/faq_small.csv" << 'EOF'
question,answer
What is the refund policy?,We offer 30-day full refunds.
How do I reset my password?,Click "Forgot password" on the login page.
Where are your servers located?,Our servers are in Frankfurt and Virginia.
EOF

# CSV sin cabecera (para probar CSV_HAS_HEADER=false)
cat > "$TEST_DIR/faq_no_header.csv" << 'EOF'
Shipping times?,Standard shipping takes 5-7 business days.
Do you ship internationally?,Yes, to 40+ countries.
EOF

# Texto plano largo (para chunking)
python3 -c "
import textwrap, random, string
paras = ['\\n\\n'.join(' '.join(random.choices(string.ascii_lowercase + ' ', k=200)) for _ in range(5)) for _ in range(10)]
print('\\n\\n'.join(paras))
" > "$TEST_DIR/long_doc.txt"

# Markdown con headers
cat > "$TEST_DIR/sample.md" << 'EOF'
# Introduction

This is the introduction section with some content.

## Section One

Details about section one including important information.

## Section Two

More details and specifics about section two.

### Subsection

Even more granular information here.
EOF

# Archivo binario (trampa)
dd if=/dev/urandom bs=1024 count=4 > "$TEST_DIR/binary_trap.bin" 2>/dev/null

# Archivo vacío (trampa)
touch "$TEST_DIR/empty_file.txt"

# Archivo con sólo whitespace
printf "   \n\t\n   " > "$TEST_DIR/whitespace_only.txt"

# CSV con filas corruptas mezcladas
cat > "$TEST_DIR/mixed_corrupt.csv" << 'EOF'
question,answer
Valid question one?,Valid answer one.
,Missing question field answer.
Only one column here
"Unclosed quote,broken csv
Valid question two?,Valid answer two.
EOF

# JSON para mutate (válido)
cat > "$TEST_DIR/mutate_valid.json" << 'EOF'
{
  "intents": [
    {
      "operation": "upsert",
      "external_id": "beta-doc-001",
      "content": "This is test document 001 for beta testing.",
      "metadata": {"source": "beta-test", "version": "1"}
    },
    {
      "operation": "upsert",
      "external_id": "beta-doc-002",
      "content": "This is test document 002 covering different topics.",
      "metadata": {"source": "beta-test", "version": "1"}
    },
    {
      "operation": "upsert",
      "external_id": "beta-doc-003",
      "content": "Third document about system reliability and uptime.",
      "metadata": {"source": "beta-test", "version": "1"}
    }
  ]
}
EOF

# JSON mutate — sólo deletions
cat > "$TEST_DIR/mutate_delete_only.json" << 'EOF'
{
  "intents": [
    {"operation": "delete", "external_id": "beta-doc-001"},
    {"operation": "delete", "external_id": "nonexistent-999"}
  ]
}
EOF

# JSON mutate — lista vacía (debe rechazarse o ser noop limpio)
cat > "$TEST_DIR/mutate_empty.json" << 'EOF'
{"intents": []}
EOF

# JSON mutate — mismo external_id upsert+delete en la misma request
cat > "$TEST_DIR/mutate_conflict.json" << 'EOF'
{
  "intents": [
    {"operation": "upsert", "external_id": "conflict-id", "content": "Some content here."},
    {"operation": "delete", "external_id": "conflict-id"}
  ]
}
EOF

# JSON mutate — contenido vacío en upsert
cat > "$TEST_DIR/mutate_empty_content.json" << 'EOF'
{
  "intents": [
    {"operation": "upsert", "external_id": "empty-content-001", "content": ""}
  ]
}
EOF

# JSONL de eval con un hit seguro y otro imposible
cat > "$TEST_DIR/eval_small.jsonl" << 'EOF'
{"question": "What is the refund policy?", "expected_doc_ids": ["beta-doc-001"], "expected_answer_contains": "refund"}
{"question": "ajsdflkajsdfkljasdflkj this cannot match anything", "expected_doc_ids": ["nonexistent"], "expected_answer_contains": "xyz"}
{"question": "Tell me about system reliability", "expected_doc_ids": ["beta-doc-003"], "expected_answer_contains": "reliability"}
EOF
```

---

## Bloque 1 — Estado inicial y bootstrap

### B1.1 — Status en DB vacía (estado cero)

```bash
SQLITE_URL=$SQLITE_URL INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-status
```

**Expectativa:** Debe arrancar sin crash. Debe reportar 0 documentos, sin índice, sin journal entries. No debe hacer ruidos en stderr sobre tablas inexistentes.

**Señal de alerta:** Excepción SQLAlchemy, traceback no capturado, mensaje de error opaco.

---

### B1.2 — Bootstrap con CSV mínimo

```bash
FAZ_CSV="$TEST_DIR/faq_small.csv" SQLITE_URL=$SQLITE_URL \
  INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-bootstrap
```

**Expectativa:** 3 documentos ingestados, log estructurado, sin prints, exit 0.

**Inspeccionar:** ¿El external_id generado sigue el esquema esperado (prefijo + contenido)? ¿Hay dedup correcto si se vuelve a ejecutar inmediatamente?

---

### B1.3 — Bootstrap idempotente (segunda ejecución)

```bash
# Ejecutar dos veces seguidas
FAZ_CSV="$TEST_DIR/faq_small.csv" SQLITE_URL=$SQLITE_URL \
  INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-bootstrap

FAZ_CSV="$TEST_DIR/faq_small.csv" SQLITE_URL=$SQLITE_URL \
  INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-bootstrap
```

**Expectativa:** Segunda ejecución = 0 nuevos documentos (dedup por chunk_dedup_sha256). Ningún error de unique constraint de SQLite.

**Señal de alerta:** Duplicados en DB, excepción UNIQUE, crash.

---

### B1.4 — Bootstrap con CSV sin cabecera (CSV_HAS_HEADER=false)

```bash
FAZ_CSV="$TEST_DIR/faq_no_header.csv" CSV_HAS_HEADER=false \
  SQLITE_URL=$SQLITE_URL INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-bootstrap
```

**Expectativa:** 2 documentos adicionales ingestados, primera columna como question.

---

### B1.5 — Bootstrap con archivo inexistente

```bash
FAZ_CSV="/nonexistent/path/file.csv" SQLITE_URL=$SQLITE_URL \
  INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-bootstrap
```

**Expectativa:** Error claro (FileNotFoundError o mensaje de usuario), exit != 0. Sin traceback crudo al usuario final.

---

## Bloque 2 — Ingest (pipeline completa)

### B2.1 — Ingest de múltiples formatos

```bash
SQLITE_URL=$SQLITE_URL INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-ingest "$TEST_DIR/long_doc.txt" "$TEST_DIR/sample.md"
```

**Expectativa:** Chunking aplicado al `.txt` largo, log con conteo de chunks por archivo. `.md` ingestado como chunks por sección.

**Verificar:** `rag-status` muestra conteo incrementado.

---

### B2.2 — Ingest de archivo vacío

```bash
SQLITE_URL=$SQLITE_URL INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-ingest "$TEST_DIR/empty_file.txt"
```

**Expectativa:** Warn/skip del archivo, exit 0 (no debe crashear). 0 documentos nuevos.

---

### B2.3 — Ingest de archivo con sólo whitespace

```bash
SQLITE_URL=$SQLITE_URL INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-ingest "$TEST_DIR/whitespace_only.txt"
```

**Expectativa:** Igual que B2.2 — skip limpio. El chunk tras limpieza de whitespace debe quedar vacío y descartarse.

---

### B2.4 — Ingest de archivo binario

```bash
SQLITE_URL=$SQLITE_URL INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-ingest "$TEST_DIR/binary_trap.bin"
```

**Expectativa:** UnicodeDecodeError capturado, log de warning, skip del archivo, exit 0.

**Señal de alerta:** Crash, o peor: contenido binario persistido como "documento".

---

### B2.5 — Ingest de directorio completo (incluyendo trampas)

```bash
SQLITE_URL=$SQLITE_URL INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-ingest "$TEST_DIR"
```

**Expectativa:** Procesa `.txt`, `.md`, `.csv` del directorio; skip de `.bin`, `.json`, `.jsonl`. Logging por archivo. Exit 0 aunque algunos fallen.

---

### B2.6 — Ingest del mismo archivo dos veces (dedup)

```bash
SQLITE_URL=$SQLITE_URL INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-ingest "$TEST_DIR/sample.md"

SQLITE_URL=$SQLITE_URL INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-ingest "$TEST_DIR/sample.md"
```

**Expectativa:** Segunda ejecución → 0 documentos nuevos (dedup por sha256). No duplicados.

---

### B2.7 — Ingest de ruta inexistente

```bash
SQLITE_URL=$SQLITE_URL INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-ingest "/nonexistent/path/to/file.txt"
```

**Expectativa:** Error claro, exit != 0.

---

## Bloque 3 — Mutate (mutation saga + journal)

### B3.1 — Mutación válida (upserts)

```bash
SQLITE_URL=$SQLITE_URL INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-mutate-docs "$TEST_DIR/mutate_valid.json"
```

**Expectativa:** 3 documentos persistidos, journal entries en estado `COMMITTED`, log de cada intent. Exit 0.

**Verificar:** `rag-status` muestra +3 documentos.

---

### B3.2 — Lista de intents vacía

```bash
SQLITE_URL=$SQLITE_URL INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-mutate-docs "$TEST_DIR/mutate_empty.json"
```

**Expectativa:** Noop limpio (0 documentos modificados) O error de validación. No debe crashear ni dejar journal en estado inconsistente.

---

### B3.3 — Contenido vacío en upsert

```bash
SQLITE_URL=$SQLITE_URL INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-mutate-docs "$TEST_DIR/mutate_empty_content.json"
```

**Expectativa:** `BadRequestError` o `UnprocessableEntityError` con mensaje claro. No debe persistir un documento con contenido vacío ni generar embeddings para texto vacío.

---

### B3.4 — Upsert y delete del mismo external_id en la misma request

```bash
SQLITE_URL=$SQLITE_URL INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-mutate-docs "$TEST_DIR/mutate_conflict.json"
```

**Expectativa:** O bien: error de validación pre-ejecución (el contrato rechaza intents conflictivos), o bien: orden determinista (upsert → delete = documento eliminado). Documentar el comportamiento real.

**Señal de alerta:** Documento queda en estado ambiguo (ni vivo ni tombstone), crash, o comportamiento no determinista entre runs.

---

### B3.5 — Delete de external_id inexistente

```bash
# Crear un JSON con sólo delete de ID no existente
cat > "$TEST_DIR/mutate_nonexistent_delete.json" << 'EOF'
{
  "intents": [
    {"operation": "delete", "external_id": "id-that-does-not-exist-99999"}
  ]
}
EOF

SQLITE_URL=$SQLITE_URL INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-mutate-docs "$TEST_DIR/mutate_nonexistent_delete.json"
```

**Expectativa:** Respuesta exitosa con `deleted_count: 0` (idempotente), O `NotFoundError` explícito. No debe crashear.

---

### B3.6 — Re-ingest de external_id tombstoneado

```bash
# Primero eliminar beta-doc-001 (del B3.1)
SQLITE_URL=$SQLITE_URL INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-mutate-docs "$TEST_DIR/mutate_delete_only.json"

# Luego intentar upsert del mismo external_id
cat > "$TEST_DIR/mutate_tombstone_reingest.json" << 'EOF'
{
  "intents": [
    {
      "operation": "upsert",
      "external_id": "beta-doc-001",
      "content": "Attempting to reingest a tombstoned document.",
      "metadata": {}
    }
  ]
}
EOF

SQLITE_URL=$SQLITE_URL INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-mutate-docs "$TEST_DIR/mutate_tombstone_reingest.json"
```

**Expectativa según CHANGELOG:** "deleting by external_id creates tombstones and blocks future re-ingest/upsert". Debe devolver `ConflictError` o similar, no silenciosamente ignorarlo.

**Señal de alerta:** El documento resucita sin advertencia, o peor, el upsert tiene éxito silencioso.

---

### B3.7 — Mutate con JSON malformado

```bash
echo '{"intents": [{"operation": "upsert"' | \
  SQLITE_URL=$SQLITE_URL INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-mutate-docs /dev/stdin
```

**Expectativa:** Error de parseo JSON claro, exit != 0.

---

## Bloque 4 — Index (rebuild + status)

### B4.1 — Rebuild en modo sparse (RETRIEVAL_MODE=sparse)

```bash
RETRIEVAL_MODE=sparse SQLITE_URL=$SQLITE_URL \
  INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-rebuild-index
```

**Expectativa:** Noop con mensaje informativo ("sparse mode does not require index rebuild"), O error controlado. No debe intentar escribir archivos FAISS en modo sparse.

---

### B4.2 — Rebuild en modo dense (RETRIEVAL_MODE=dense) con documentos existentes

```bash
RETRIEVAL_MODE=dense SQLITE_URL=$SQLITE_URL \
  INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-rebuild-index
```

**Expectativa:** FAISS index reconstruido desde SQLite, id_map.json actualizado, log de progreso con batch size, exit 0.

**Verificar:** Los archivos `faiss.index` y `id_map.json` existen y tienen tamaño > 0.

---

### B4.3 — Rebuild en DB vacía (sin documentos)

```bash
export EMPTY_DIR=$(mktemp -d /tmp/rag-empty-XXXXXX)
RETRIEVAL_MODE=dense SQLITE_URL="sqlite:///$EMPTY_DIR/empty.db" \
  INDEX_PATH="$EMPTY_DIR/faiss.index" ID_MAP_PATH="$EMPTY_DIR/id_map.json" \
  uv run rag-rebuild-index
```

**Expectativa:** Graceful handling — "0 documents, nothing to index" o similar. No debe crear un FAISS index vacío y luego romperse en queries posteriores.

---

### B4.4 — Status post-rebuild

```bash
RETRIEVAL_MODE=dense SQLITE_URL=$SQLITE_URL \
  INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-status
```

**Expectativa:** `ntotal` del índice FAISS coincide con el número de documentos en SQLite (no > no <). Sin drift detectado.

---

### B4.5 — Drift detection: ingest sin rebuild

```bash
# Ingest nuevo documento sin reconstruir el índice
cat > "$TEST_DIR/new_doc.txt" << 'EOF'
This is a brand new document that was ingested after the last index rebuild.
It should appear as a drift between SQL document count and FAISS index total.
EOF

SQLITE_URL=$SQLITE_URL INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-ingest "$TEST_DIR/new_doc.txt"

RETRIEVAL_MODE=dense SQLITE_URL=$SQLITE_URL \
  INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-status
```

**Expectativa:** `rag-status` detecta drift (SQL count > FAISS ntotal) y lo reporta como advertencia. No debe crashear ni silenciarlo.

---

## Bloque 5 — Eval pipeline

### B5.1 — Eval con dataset pequeño (sparse mode)

```bash
# Asegurarse de que los docs del eval estén ingestados
SQLITE_URL=$SQLITE_URL INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-mutate-docs "$TEST_DIR/mutate_valid.json"

RETRIEVAL_MODE=sparse SQLITE_URL=$SQLITE_URL \
  INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-eval --dataset "$TEST_DIR/eval_small.jsonl"
```

**Expectativa:** Hit rate y MRR reportados. El segundo query (imposible de matchear) debe bajar el hit rate. Exit 0 aunque hit rate sea 0.

---

### B5.2 — Eval con dataset del repo (golden)

```bash
RETRIEVAL_MODE=sparse SQLITE_URL=$SQLITE_URL \
  INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-eval --dataset datasets/rag_eval_v1.jsonl
```

**Expectativa:** Resultados coherentes. Comparar vs. baseline documentado (si existe).

---

### B5.3 — Eval con JSONL malformado

```bash
cat > "$TEST_DIR/eval_corrupt.jsonl" << 'EOF'
{"question": "Valid question?", "expected_doc_ids": ["doc1"]}
{invalid json here
{"question": "Another valid?", "expected_doc_ids": ["doc2"]}
EOF

RETRIEVAL_MODE=sparse SQLITE_URL=$SQLITE_URL \
  uv run rag-eval --dataset "$TEST_DIR/eval_corrupt.jsonl"
```

**Expectativa:** Salta la línea corrupta con warning, procesa las válidas, reporta métricas parciales. No debe crashear.

---

### B5.4 — Eval con dataset vacío

```bash
touch "$TEST_DIR/eval_empty.jsonl"
RETRIEVAL_MODE=sparse SQLITE_URL=$SQLITE_URL \
  uv run rag-eval --dataset "$TEST_DIR/eval_empty.jsonl"
```

**Expectativa:** "0 records evaluated" con mensaje claro. No división por cero en métricas.

---

## Bloque 6 — HTTP API (servidor levantado)

> Iniciar servidor en terminal separada:
> ```bash
> RETRIEVAL_MODE=sparse SQLITE_URL=$SQLITE_URL INDEX_PATH=$INDEX_PATH \
>   ID_MAP_PATH=$ID_MAP_PATH DEBUG=true \
>   uv run rag-server --host 127.0.0.1 --port 8765
> ```

```bash
export API_BASE="http://127.0.0.1:8765/api"
```

### B6.1 — Health y readiness básica

```bash
curl -s "$API_BASE/health" | python3 -m json.tool
curl -s "$API_BASE/ready" | python3 -m json.tool
curl -s "$API_BASE/health/ollama" | python3 -m json.tool
```

**Expectativa:**
- `/health`: `{"status": "ok"}` siempre que la DB sea accesible.
- `/ready`: Puede reportar Ollama como no disponible (si no está corriendo), pero el campo debe existir y el estado debe ser legible.
- `/health/ollama`: `{"available": false}` si Ollama no está corriendo — no debe crashear.

---

### B6.2 — Config y templates

```bash
curl -s "$API_BASE/config" | python3 -m json.tool
curl -s "$API_BASE/templates" | python3 -m json.tool
```

**Expectativa:** JSON válido con los campos esperados. `retrieval_mode: "sparse"` refleja la config del entorno.

---

### B6.3 — Ask con Ollama sin correr

```bash
curl -s -X POST "$API_BASE/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?", "k": 3}' | python3 -m json.tool
```

**Expectativa:** Si Ollama no está disponible, debe devolver HTTP 503 con mensaje claro. No un 500 con traceback en el body. Los retrieved documents pueden mostrarse aunque la generación falle (si el sistema lo soporta).

---

### B6.4 — Ask con pregunta vacía

```bash
curl -s -X POST "$API_BASE/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "", "k": 3}' | python3 -m json.tool
```

**Expectativa:** HTTP 400/422 con mensaje de validación. No 500.

---

### B6.5 — Ask con k=0 y k negativo

```bash
curl -s -X POST "$API_BASE/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Test", "k": 0}' | python3 -m json.tool

curl -s -X POST "$API_BASE/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Test", "k": -1}' | python3 -m json.tool
```

**Expectativa:** HTTP 422 de Pydantic, no crash.

---

### B6.6 — Ask con k > total de documentos

```bash
curl -s -X POST "$API_BASE/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Test reliability", "k": 9999}' | python3 -m json.tool
```

**Expectativa:** Responde con todos los documentos disponibles (k clipped a ntotal) sin error. O error controlado si FAISS no admite k > ntotal.

---

### B6.7 — List docs y history paginados

```bash
# Primera página
curl -s "$API_BASE/docs?limit=2&offset=0" | python3 -m json.tool
# Segunda página
curl -s "$API_BASE/docs?limit=2&offset=2" | python3 -m json.tool
# Límite 0 (borde)
curl -s "$API_BASE/docs?limit=0" | python3 -m json.tool
# Límite negativo
curl -s "$API_BASE/docs?limit=-1" | python3 -m json.tool

curl -s "$API_BASE/history?limit=10&offset=0" | python3 -m json.tool
```

**Expectativa:** Paginación coherente, `limit=0` y `limit=-1` son 422 o devuelven lista vacía (documentar comportamiento real).

---

### B6.8 — Docs mutate via API

```bash
# Upsert válido
curl -s -X POST "$API_BASE/docs/mutate" \
  -H "Content-Type: application/json" \
  -d @"$TEST_DIR/mutate_valid.json" | python3 -m json.tool

# Lista vacía
curl -s -X POST "$API_BASE/docs/mutate" \
  -H "Content-Type: application/json" \
  -d '{"intents": []}' | python3 -m json.tool

# Contenido vacío
curl -s -X POST "$API_BASE/docs/mutate" \
  -H "Content-Type: application/json" \
  -d '{"intents": [{"operation": "upsert", "external_id": "x", "content": ""}]}' | python3 -m json.tool
```

---

### B6.9 — Import de archivo via API (multipart)

```bash
# TXT válido
curl -s -X POST "$API_BASE/docs/import" \
  -F "file=@$TEST_DIR/sample.md;type=text/markdown" | python3 -m json.tool

# Archivo vacío
curl -s -X POST "$API_BASE/docs/import" \
  -F "file=@$TEST_DIR/empty_file.txt;type=text/plain" | python3 -m json.tool

# Archivo binario
curl -s -X POST "$API_BASE/docs/import" \
  -F "file=@$TEST_DIR/binary_trap.bin;type=application/octet-stream" | python3 -m json.tool

# Archivo demasiado grande (generar 11MB > límite típico de 10MB)
dd if=/dev/urandom bs=1024 count=11264 2>/dev/null | base64 > "$TEST_DIR/too_large.txt"
curl -s -X POST "$API_BASE/docs/import" \
  -F "file=@$TEST_DIR/too_large.txt;type=text/plain" | python3 -m json.tool
```

**Expectativa:** `too_large.txt` → HTTP 413/422 `PayloadTooLargeError`. Binario → 422 con mensaje de unsupported format o decode error.

---

### B6.10 — Index rebuild via API

```bash
# En modo sparse (debe ser noop o error controlado)
curl -s -X POST "$API_BASE/index/rebuild" | python3 -m json.tool
```

---

### B6.11 — Autenticación (API_KEY enforcement)

```bash
# Reiniciar servidor con API_KEY configurado
# CTRL+C el servidor anterior, luego:
# API_KEY=supersecret RETRIEVAL_MODE=sparse SQLITE_URL=$SQLITE_URL \
#   INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
#   uv run rag-server --host 127.0.0.1 --port 8765

# Sin key → 401/403
curl -s "$API_BASE/health" | python3 -m json.tool
curl -s "$API_BASE/ask" -X POST \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "k": 1}' | python3 -m json.tool

# Con key incorrecta → 401/403
curl -s "$API_BASE/ask" -X POST \
  -H "X-API-Key: wrongkey" \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "k": 1}' | python3 -m json.tool

# Con key correcta → funciona
curl -s "$API_BASE/ask" -X POST \
  -H "X-API-Key: supersecret" \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "k": 1}' | python3 -m json.tool
```

**Expectativa:** Sin key o key incorrecta → HTTP 401 con body JSON, nunca 500. `/health` puede ser pública (verificar).

---

### B6.12 — Concurrent mutations (race condition)

```bash
# 5 requests de mutate concurrentes con external_ids distintos
for i in $(seq 1 5); do
  cat > "$TEST_DIR/concurrent_$i.json" << EOF
{
  "intents": [
    {
      "operation": "upsert",
      "external_id": "concurrent-doc-$i",
      "content": "Concurrent document number $i for race condition testing.",
      "metadata": {"batch": "$i"}
    }
  ]
}
EOF
done

# Lanzar en paralelo
for i in $(seq 1 5); do
  curl -s -X POST "$API_BASE/docs/mutate" \
    -H "Content-Type: application/json" \
    -d @"$TEST_DIR/concurrent_$i.json" &
done
wait

# Verificar que los 5 documentos existen y no hay duplicados
curl -s "$API_BASE/docs?limit=50" | python3 -m json.tool | grep -c "concurrent-doc"
```

**Expectativa:** Los 5 documentos existen exactamente 1 vez cada uno. No hay deadlocks ni errores de lock acquisition. Journal en estado COMMITTED para todos.

---

### B6.13 — OpenRouter proxy (sin API key)

```bash
curl -s -X POST "$API_BASE/openrouter/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }' | python3 -m json.tool
```

**Expectativa:** HTTP 503 o 502 con mensaje claro sobre OPENROUTER_API_KEY no configurado. No un timeout de 30s, no un 500 genérico.

---

## Bloque 7 — Recovery y resiliencia

### B7.1 — Arranque con archivos FAISS corruptos

```bash
# Corromper el índice FAISS existente
echo "THIS IS NOT A FAISS INDEX" > $INDEX_PATH

RETRIEVAL_MODE=dense SQLITE_URL=$SQLITE_URL \
  INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-status
```

**Expectativa:** Error detectado en startup/status, mensaje de "index corrupted or missing, rebuild required". No crash, no silencio.

---

### B7.2 — Arranque con id_map.json malformado

```bash
echo '{"this_is": "not a valid id map"}' > $ID_MAP_PATH

RETRIEVAL_MODE=dense SQLITE_URL=$SQLITE_URL \
  INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-status
```

**Expectativa:** Error capturado, solicitud de rebuild o mensaje claro. No IndexError al acceder a campos del id_map.

---

### B7.3 — id_map.json vacío vs. index FAISS válido (mismatch)

```bash
# Reconstruir index limpio primero
RETRIEVAL_MODE=dense SQLITE_URL=$SQLITE_URL \
  INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-rebuild-index

# Luego vaciar el id_map
echo '{}' > $ID_MAP_PATH

# Intentar un ask
curl -s -X POST "$API_BASE/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "refund policy", "k": 3}' | python3 -m json.tool
```

**Expectativa:** Error de validación de consistencia (ids/embeddings length mismatch), HTTP 503 o mensaje de "index requires rebuild". No `IndexError: list index out of range`.

---

### B7.4 — Simular crash mid-mutation via journal

```bash
# Verificar que el journal recovery funciona si hay entradas en PREPARED/IN_PROGRESS
# (Requiere inspección directa de la DB si el CLI expone esto)
RETRIEVAL_MODE=sparse SQLITE_URL=$SQLITE_URL \
  INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-status
```

**Buscar en el log:** Mensajes de `mutation_recovery` al arrancar. Verificar que las entradas en estado `PREPARED` se resetean con warning (no silenciosamente, no crash).

---

## Bloque 8 — Configuración y settings edge cases

### B8.1 — Variables de entorno inválidas

```bash
# Puerto inválido
APP_PORT=99999 uv run rag-server 2>&1 | head -5

# Temperatura fuera de rango
OPENAI_TEMPERATURE=5.0 RETRIEVAL_MODE=sparse SQLITE_URL=$SQLITE_URL \
  uv run rag-status 2>&1 | head -5

# Modo de retrieval inválido
RETRIEVAL_MODE=unicorn SQLITE_URL=$SQLITE_URL \
  uv run rag-status 2>&1 | head -5
```

**Expectativa:** Validación de Pydantic Settings en startup, mensaje claro de qué campo es inválido, exit != 0.

---

### B8.2 — SQLITE_URL apuntando a directorio sin permisos de escritura

```bash
SQLITE_URL="sqlite:////root/no_access/test.db" \
  uv run rag-status 2>&1 | head -10
```

**Expectativa:** Error de permisos capturado con mensaje claro, no traceback de SQLAlchemy crudo.

---

### B8.3 — INGEST_BATCH_SIZE extremos

```bash
# Batch size 1 (ingest muy lento pero correcto)
INGEST_BATCH_SIZE=1 SQLITE_URL=$SQLITE_URL \
  INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-ingest "$TEST_DIR/sample.md"

# Batch size muy grande (mayor que el total de docs)
INGEST_BATCH_SIZE=10000 SQLITE_URL=$SQLITE_URL \
  INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  uv run rag-ingest "$TEST_DIR/sample.md"
```

**Expectativa:** Ambos terminan correctamente. El batch size 1 puede ser lento, pero no debe crashear.

---

## Bloque 9 — Frontend / UI (si aplica)

```bash
# Acceder a la UI embedded en el servidor
curl -s "http://127.0.0.1:8765/" | head -20
```

**Verificar:**
- Responde con HTML (no 404, no 500)
- La UI carga sin errores de JS en consola del navegador
- El campo de pregunta envía al endpoint correcto
- Con Ollama caído: la UI muestra error claro, no spinner infinito

---

## Checklist de señales de alerta críticas

Antes de dar el release por válido, confirmar que **ninguno** de estos patrones aparece:

| # | Señal de alerta | Dónde verificar |
|---|----------------|-----------------|
| A1 | Traceback de Python en stdout/stderr para inputs de usuario | Todos los tests de inputs inválidos |
| A2 | `print()` en producción (en lugar de logger) | Logs de cualquier comando |
| A3 | `assert` en código de producción (AssertionError en vez de excepción explícita) | Inputs extremos |
| A4 | HTTP 500 para inputs de usuario inválidos (debe ser 400/422) | B6.4, B6.5, B6.9 |
| A5 | Duplicados en DB después de doble-ingest | B1.3, B2.6 |
| A6 | Documento resucitado después de tombstone | B3.6 |
| A7 | FAISS ntotal != SQL count después de rebuild | B4.4 |
| A8 | Deadlock o hang en mutations concurrentes | B6.12 |
| A9 | División por cero en métricas de eval | B5.4 |
| A10 | Journal en estado PREPARED/IN_PROGRESS tras restart limpio | B7.4 |
| A11 | Timeout sin mensaje de error para providers caídos | B6.3, B6.13 |
| A12 | Contenido binario persistido como documento | B2.4 |

---

## Registro de resultados

```markdown
| Test | Resultado | Comportamiento real | Veredicto |
|------|-----------|--------------------| --------- |
| B1.1 |           |                    |           |
| B1.2 |           |                    |           |
...
```

**Veredictos posibles:** `PASS` / `FAIL` / `WARN` (comportamiento incorrecto pero no crítico) / `SKIP` (requires external service)

---

## Escenarios de regresión específicos del CHANGELOG v1.3.0

Basados en los cambios declarados de esta release:

```bash
# R1: Verificar que alchemy_engine.py NO existe (breaking change)
ls src/local_rag_backend/infrastructure/persistence/sql/alchemy_engine.py 2>&1
# Esperado: "No such file or directory"

# R2: Verificar path de composición unificado
python3 -c "
from local_rag_backend.composition.container import AppContainer
from local_rag_backend.settings import Settings
s = Settings(_env_file=None)
c = AppContainer.from_settings(s)
print('OK: from_settings() works')
"

# R3: Verificar que runtime_wiring_defaults() existe y es callable
python3 -c "
from local_rag_backend.composition.container import AppContainer
from local_rag_backend.settings import Settings
s = Settings(_env_file=None)
c = AppContainer.from_settings(s)
defaults = c.runtime_wiring_defaults()
print(f'OK: runtime_wiring_defaults() returned {type(defaults)}')
"

# R4: Verificar que bootstrap va por MutationCoordinator (no legacy ETL)
# Inspectar el log de rag-bootstrap buscando "mutation" en vez de "etl"
FAZ_CSV="$TEST_DIR/faq_small.csv" SQLITE_URL=$SQLITE_URL \
  INDEX_PATH=$INDEX_PATH ID_MAP_PATH=$ID_MAP_PATH \
  LOG_LEVEL=debug uv run rag-bootstrap 2>&1 | grep -i "mutation\|etl\|coordinator"
```

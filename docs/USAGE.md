# RAG Stateful: mutación canónica y evaluación

Este documento describe el uso avanzado de `rag-prototype` como librería de Python para construir flujos de trabajo de RAG (Retrieval-Augmented Generation) personalizados. El quick start vive en `README.md`; aquí se documentan la configuración, la ingesta canónica, la consulta y la evaluación.

## Por qué existe esta complejidad

Este proyecto no está optimizado para "subir documentos y preguntar". Está diseñado como una plataforma RAG stateful donde el estado canónico, la proyección de búsqueda y la evaluación viven separados por contrato.

Por eso existen tres piezas que no conviene saltarse:

* `MutationCoordinator`: garantiza que toda mutación pase por un write-path canónico, con saga durable o path atómico según el backend.
* `rag-rebuild-index` / `POST /api/index/rebuild`: el índice de lectura es reparable y derivado; el rebuild explícito evita que el estado corrupto se oculte como si fuera normal.
* `rag-eval` y `rag-eval-compare`: cualquier cambio de chunking, retrieval o reranking debe pasar por un gate reproducible para evitar regresiones silenciosas.

Si tu caso de uso no necesita este nivel de control, probablemente te baste una topología más simple. Si sí lo necesitas, esta complejidad es intencional.

## Requisitos Previos

1.  **Ollama en ejecución**: Asegúrate de tener Ollama instalado y un modelo descargado (ej. `ollama pull lfm2.5-thinking`).
2.  **Entorno listo**: Este documento asume que el proyecto ya está instalado y configurado. Los pasos de instalación viven en el `README.md`.

---

## Paso 1: Configuración del Entorno

La librería se configura mediante un único archivo `config.yaml` en la raíz del proyecto.
Las rutas se resuelven relativas a ese archivo, no al directorio actual.
Toma como base `config.example.yaml` y copia el archivo a `config.yaml` antes de editarlo.
Importante: `docker-compose.yml` puede definir variables de entorno para el contenedor, pero el runtime no las usa como fuente de configuración hoy. Si despliegas con Compose, mantén `config.yaml` sincronizado o móntalo explícitamente.

```yaml
# config.yaml
persistence_backend: local_split
app_host: 127.0.0.1
app_port: 8000
debug: false
log_level: INFO
enable_monitoring: false
enable_reranker: false

api_key: null
public_bind_requires_api_key: true
cors_allow_origins: []

search_backend: local_split
retrieval_mode: sparse
vector_backend: auto
hybrid_retrieval_alpha: 0.5
dual_candidate_k: 50
st_embedding_model: all-MiniLM-L6-v2

openai_api_key: null
ollama_enabled: true
ollama_model: lfm2.5-thinking
sqlite_url: sqlite:///./data/custom_app.db
data_dir: data
faq_csv: data/faq.csv
eval_dataset_path: datasets/rag_eval_v1.jsonl
```

Nota de topología: `search_backend` controla el motor de consulta independientemente de
`persistence_backend`. Consulta la matriz del README para validar `dense`, `dual` o `hybrid`
antes de cambiar valores.

## Paso 2: Implementación de un `Loader` Personalizado

La librería define una interfaz (`port`) para los cargadores de datos. Para crear uno custom, solo es necesario implementar la clase `LoaderPort`.


```python
# my_custom_loader.py

from typing import Iterable, Any

from local_rag_backend.core.domain.entities import LoadedItem
from local_rag_backend.core.domain.types import ItemLineage
from local_rag_backend.core.ports import LoaderPort

class DictListLoader(LoaderPort):
    """Un cargador personalizado que lee datos de una lista de diccionarios."""

    def __init__(self, data: list[dict[str, Any]]):
        self._data = data

    def load(self) -> Iterable[LoadedItem]:
        """Genera LoadedItems a partir de la lista de datos."""
        for i, item in enumerate(self._data):
            # Asume que cada diccionario tiene 'title' y 'content'
            text = f"{item.get('title', '')}\n\n{item.get('content', '')}"
            metadata = {"source": f"dict_item_{i}", **item.get('metadata', {})}
            yield LoadedItem(
                text=text.strip(),
                lineage=ItemLineage(
                    source_uri=f"dict://item/{i}",
                    loader_name="DictListLoader",
                ),
                metadata=metadata,
            )

```

## Paso 3: Ingesta canónica con `AppContainer` + `MutationCoordinator`

Para persistir datos usa siempre el write-path canónico. El script siguiente convierte los items del `Loader` en `MutationIntent` y delega en `MutationCoordinator`; no abre SQLite ni escribe a mano en repositorios concretos.

```python
# run_ingestion.py

from local_rag_backend.composition.container import AppContainer
from local_rag_backend.composition.adapters import build_dense_embedder_from_settings
from local_rag_backend.core.use_cases.docs_mutation import (
    MutationCoordinator,
    MutationIntent,
    MutationUpsertInput,
)
from local_rag_backend.settings import settings

# 1. Importar Loader (custom)
from my_custom_loader import DictListLoader

# 2. Datos de ejemplo
my_data = [
    {"title": "Inteligencia Artificial", "content": "La IA es la simulación de procesos de inteligencia humana.", "metadata": {"category": "Tech"}},
    {"title": "Hexagonal Architecture", "content": "Es un patrón de diseño de software que desacopla el núcleo de la aplicación.", "metadata": {"category": "Software"}}
]

def main():
    print("--- Iniciando ingesta canónica ---")

    # 3. Instanciar el contenedor y el coordinador canónico
    container = AppContainer.from_settings(settings)
    coordinator = MutationCoordinator(
        settings_obj=settings,
        ports=container.docs_mutation_ports(
            build_embedder=lambda: build_dense_embedder_from_settings(settings_obj=settings),
        ),
    )
    custom_loader = DictListLoader(data=my_data)

    # 4. Convertir cada LoadedItem a MutationUpsertInput
    upserts: list[MutationUpsertInput] = []
    for i, item in enumerate(custom_loader.load()):
        upserts.append(
            MutationUpsertInput(
                external_id=f"dict_item_{i}",
                content=item.text,
                source_id=item.lineage.source_uri,
                metadata=item.metadata,
            )
        )

    # 5. Ejecutar la mutación canónica
    summary = coordinator.execute(
        MutationIntent(
            op_id="script-op-1",
            upserts=tuple(upserts),
            source="script:custom_loader",
        )
    )
    print(f"\n[OK] Ingesta completada: {summary}")

if __name__ == "__main__":
    main()

```

Ejecuta el script para poblar tu backend canónico:

```bash
python run_ingestion.py
```

Nota: el ejemplo evita escrituras directas a SQLite. El mismo flujo funciona en `sparse`, `dense` y `hybrid`; el coordinador decide la estrategia adecuada según la configuración.

## Paso 4: Script de Consulta con Ollama

Script para hacer preguntas a los datos utilizando el `RagService` y Ollama.

```python
# run_query.py

from local_rag_backend.bootstrap import bootstrap_rag_service

def main():
    print("--- Iniciando servicio RAG para consulta ---")

    # La función bootstrap_rag_service crea y conecta todos los componentes
    # necesarios para realizar consultas (repositorios, retrievers, generador).
    rag_service = bootstrap_rag_service()

    question = "¿Qué es la arquitectura hexagonal?"
    print(f"\nPregunta: {question}")

    # Realizar la consulta
    response = rag_service.ask(question)

    print(f"\nRespuesta de Ollama:\n{response['answer']}")

    print("\n--- Fuentes utilizadas ---")
    for doc, score in zip(response["docs"], response["scores"], strict=False):
        print(f"- ID: {doc.id}, Score: {score:.3f}, Contenido: {doc.content[:100]}...")

if __name__ == "__main__":
    main()

```

Ejecuta este script para obtener una respuesta:

```bash
python run_query.py
```

> Siguiendo estos pasos, puedes adaptar este proyecto para entender cómo construir un sistema RAG, con soporte para modelos locales con Ollama.

---

## CLI: ingesta desde ficheros/directorios (txt/md/csv)

Para un flujo rápido sin escribir código, puedes ingestar desde rutas locales:

```bash
# Ingesta desde un fichero o un directorio (recursivo por defecto)
rag-ingest ./docs ./notas.md ./data/faq.csv

# Ver qué se procesaría sin escribir en SQLite/índice vectorial
rag-ingest --dry-run ./docs
```

Notas:

* En `local_split` + `dense`/`hybrid`, la CLI actualiza SQLite y el índice vectorial local de forma consistente (FAISS o NumPy, según `VECTOR_BACKEND`) y borra chunks obsoletos si un fichero se acorta.
* En `elasticsearch` + `dense`/`hybrid`, la CLI usa el backend unificado: documentos, embeddings, history, system state y tombstones viven en Elasticsearch.
* La detección de formato es best-effort (no solo extensión). Opcionalmente puedes instalar `python-magic` con el extra `magic`.
* Si no quieres seguir enlaces simbólicos (incluyendo rutas raíz que sean symlink), usa `--no-follow-symlinks`.

---

## Mantenimiento (dense/dual/hybrid): mutación canónica + repair explícito

`MutationCoordinator` es el write-path canónico en ambos backends:

* `local_split`: `DURABLE_SAGA` con journal duradero, SQLite como store canónico y vector index local como estado derivado.
* `elasticsearch`: path atómico sobre backend unificado; no hay journal de mutación local ni lock SQL.

### 1) CLI (canónico)

```bash
# Upsert (idempotente por op_id)
cat > /tmp/mutate_upsert.json <<'JSON'
{"op_id":"op-upsert-1","upserts":[{"external_id":"doc-1","content":"hola"}]}
JSON
rag-mutate-docs --json /tmp/mutate_upsert.json

# Delete por IDs SQL
cat > /tmp/mutate_delete_ids.json <<'JSON'
{"op_id":"op-del-ids-1","delete_ids":["doc:...","doc:..."]}
JSON
rag-mutate-docs --json /tmp/mutate_delete_ids.json

# Delete por external_id (crea tombstones)
cat > /tmp/mutate_delete_ext.json <<'JSON'
{"op_id":"op-del-ext-1","delete_external_ids":["chunk:<sha256>","file:/abs/path:part=file:chunk=0"]}
JSON
rag-mutate-docs --json /tmp/mutate_delete_ext.json

# Import/sync canónico para productores externos (p.ej. RepoGPT)
cat > /tmp/canonical_import.json <<'JSON'
{
  "scope":"repogpt:demo",
  "snapshot_id":"snap-1",
  "replace_scope": true,
  "documents":[
    {
      "external_id":"repogpt:demo:1",
      "source_id":"repogpt:demo:file:src/app.py",
      "content":"def hello():\n    return 1\n",
      "metadata":{"path":"src/app.py","unit_type":"function"}
    }
  ]
}
JSON
rag-import-canonical --json /tmp/canonical_import.json

# Repair explícito del estado de retrieval
rag-rebuild-index
```

### 2) API (FastAPI)

Levanta el servidor antes de llamar a la API:

```bash
rag-server
```

Probes operacionales (públicas):

```bash
curl -s http://localhost:8000/healthz
curl -s http://localhost:8000/readyz
```

`/healthz` solo valida disponibilidad básica del servicio. `/readyz` es más estricto y puede devolver
`503` si no hay proveedor LLM configurado (`openai_api_key`, `ollama_enabled: true`, u OpenRouter con
`openrouter_enabled: true` + `openrouter_api_key`), aunque la app y SQLite estén sanos.

```bash
curl -X POST "http://localhost:8000/api/docs/mutate" \
  -H "Content-Type: application/json" \
  -d '{"op_id":"op-1","upserts":[{"external_id":"doc-1","content":"hola"}]}'

curl -X POST "http://localhost:8000/api/docs/mutate" \
  -H "Content-Type: application/json" \
  -d '{"op_id":"op-2","delete_external_ids":["chunk:<sha256>"]}'

curl -X POST "http://localhost:8000/api/docs/import-canonical" \
  -H "Content-Type: application/json" \
  -d '{"scope":"repogpt:demo","snapshot_id":"snap-1","replace_scope":true,"documents":[{"external_id":"repogpt:demo:1","source_id":"repogpt:demo:file:src/app.py","content":"def hello():\n    return 1\n","metadata":{"path":"src/app.py","unit_type":"function"}}]}'

curl -X POST "http://localhost:8000/api/docs/query" \
  -H "Content-Type: application/json" \
  -d '{"limit":50,"offset":0,"filters":[{"field":"scope","values":["repogpt:demo"]},{"field":"metadata.unit_type","values":["function"]}]}'

curl -X POST "http://localhost:8000/api/index/rebuild"
```

Si has configurado `api_key` en `config.yaml`, añade `-H "X-API-Key: <api_key>"`. Además, por defecto las peticiones no-locales requieren API key.

### 3) Como librería (flujo programático recomendado)

```python
from local_rag_backend.core.use_cases.docs_mutation import (
    MutationCoordinator,
    MutationIntent,
    MutationUpsertInput,
)
from local_rag_backend.composition.adapters import build_dense_embedder_from_settings
from local_rag_backend.composition.container import AppContainer
from local_rag_backend.settings import settings

container = AppContainer.from_settings(settings)
ports = container.docs_mutation_ports(
    build_embedder=lambda: build_dense_embedder_from_settings(settings_obj=settings),
)
coordinator = MutationCoordinator(settings_obj=settings, ports=ports)

summary = coordinator.execute(
    MutationIntent(
        op_id="script-op-1",
        upserts=(MutationUpsertInput(external_id="doc-1", content="hola"),),
        delete_external_ids=("chunk:<sha256>",),
        source="script:custom",
    )
)
print(summary)
```

Notas:

* En `local_split`, el rebuild recompone el índice vectorial local desde el store canónico.
* En `elasticsearch`, el rebuild re-embebe los documentos del índice de documentos y actualiza los vectores in-place.
* El rebuild completo queda para reparación explícita (`rag-rebuild-index` / `POST /api/index/rebuild`), no como fallback normal de mutación.
* `rag-import-canonical` / `POST /api/docs/import-canonical` hacen sync por `scope + snapshot_id`; con `replace_scope=true` eliminan documentos obsoletos sin crear tombstones. Si omites `replace_scope`, CLI y HTTP ahora usan el mismo default: `true`.
* CLI, HTTP y MCP comparten la misma validación tipada para canonical import; `RepoGPT code-units v4` se normaliza en el borde de transporte, no en el core del importador.
* Los filtros públicos soportados son sólo `scope`, `snapshot_id`, `source_id` y `metadata.<key>`.
* `rag_status` devuelve un `runtime` estructurado con topología, seguridad, backends y rutas, además de `health` e `index` cuando aplican.

### RepoGPT contract

`RepoGPT code-units v4` es el contrato soportado para la integración canónica de código:

* `kind = "code-units"`
* `schema_version = "4"`
* `scope`, `snapshot_id`, `replace_scope`
* `documents[]` con `external_id`, `content`, `metadata`

El import canónico sigue siendo genérico; no se especializa el use case al dominio RepoGPT. La validación específica vive en el borde de transporte para detectar payloads `code-units` desalineados antes de tocar el write-path canónico.

---

## Operabilidad: métricas, evaluación y reranker

Monitoring mínimo (Prometheus):
Si activas `enable_monitoring: true` en `config.yaml`, `rag-server` expone `/metrics` cuando la
dependencia opcional está instalada.

Si tienes `api_key` en `config.yaml`, añade `-H "X-API-Key: <api_key>"` al `curl`.

Evaluación offline reproducible (gate):

```bash
rag-eval --retrieval-mode sparse
rag-eval --retrieval-mode dense --candidate-k 20
rag-eval --retrieval-mode dual --dual-candidate-k 50
rag-eval --retrieval-mode hybrid --hybrid-alpha 0.5
rag-eval-compare --candidate-mode dual --candidate-dual-candidate-k 50
rag-eval-batch --specs /tmp/rag-eval-batch-specs.json
```

Dataset por defecto: `datasets/rag_eval_v1.jsonl` (o `eval_dataset_path` en `config.yaml`).
El comando reporta métricas estándar de IR a `@k` (`nDCG`, `MAP`, `MRR`, `P`, `Recall`).
La evaluación usa un runtime local aislado bajo `<data_dir>/_eval_workspaces/`; no reutiliza ni muta el índice principal.
Ese runtime sí puede reutilizar un índice denso de evaluación ya persistido cuando coinciden:

* la firma del dataset (`dataset_id` + documentos)
* el conjunto de `doc_ids` cargados en el workspace
* el backend vectorial y el manifest esperado del índice

Si cambias el modelo de embeddings, el backend vectorial o cualquier input del manifest denso, la evaluación invalida ese workspace y reconstruye el índice aislado.
El rebuild denso se hace en batches acotados para reducir picos de memoria en corpora grandes, pero sigue siendo un rebuild completo del workspace de evaluación cuando hay drift.
El dataset se valida de forma estricta: IDs duplicados, relevantes vacíos o relevantes fuera del corpus fallan al cargar.
Los overrides de modo son explícitos:
- `--candidate-k` sólo para `dense`
- `--dual-candidate-k` sólo para `dual`
- `--hybrid-alpha` sólo para `hybrid`

Override útil para wrappers/benchmarks:

```bash
RAG_PERF_METRICS_OUT=/tmp/rag-perf.json rag-eval --retrieval-mode dense
```

Ese env var sobrescribe `perf_metrics_out_path` en tiempo de carga de settings sin editar `config.yaml`.

Comparación baseline-vs-candidate (“Detector de Placebo RAG”):

```bash
rag-eval-compare \
  --candidate-mode dual \
  --candidate-dual-candidate-k 50 \
  --min-delta-ndcg 0.02 \
  --min-delta-map 0.02 \
  --min-delta-mrr 0.02 \
  --max-regression-precision 0.01 \
  --max-regression-recall 0.01 \
  --json-out /tmp/rag-eval-compare.json
```

Comportamiento:
- baseline por defecto: `sparse` sin reranker
- candidate: la configuración que quieras validar
- exit code `0`: pasa el gate
- exit code `1`: la candidate no mejora lo suficiente o degrada métricas críticas
- exit code `2`: error de configuración, dependencia o entorno

El JSON de salida incluye:
- `baseline`
- `candidate`
- `delta`

Smoke e2e reproducible:

```bash
bash scripts/test_rag_eval_compare_e2e.sh
```

Reranker opcional (mejora de calidad medible con `rag-eval`):

```bash
# set these in config.yaml:
# enable_reranker: true
# reranker_candidate_k: 20
```

Las notas internas de `synergy` y los packs de demo/evaluación específicos de ese workspace
se movieron a [`docs/internal-synergy.md`](./internal-synergy.md) para mantener esta guía
centrada en el flujo de uso avanzado del proyecto.

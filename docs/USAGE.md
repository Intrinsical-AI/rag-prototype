# Guía de Uso Avanzado: Orquestación de RAG Local con Ollama

Este documento describe cómo utilizar `rag-prototype` como una librería de Python para construir flujos de trabajo de RAG (Retrieval-Augmented Generation) personalizados. Aprenderás a implementar tu propio cargador de datos (`Loader`) y a orquestar el proceso de ingesta y consulta utilizando un modelo local de Ollama.

## Requisitos Previos

1.  **Ollama en ejecución**: Asegúrate de tener Ollama instalado y un modelo descargado (ej. `ollama pull lfm2.5-thinking`).
2.  **Proyecto instalado**: Instala el proyecto en modo editable para facilitar el desarrollo:

    ```bash
    uv venv .venv
    source .venv/bin/activate
    # Windows: .venv\Scripts\activate
    uv sync --frozen
    ```

3.  **Extras según el flujo** (opcionales):

    ```bash
    # Si vas a usar API HTTP / rag-server
    uv sync --frozen --extra server

    # Si además quieres endpoint /metrics (Prometheus)
    uv sync --frozen --extra server --extra monitoring
    ```

---

## Paso 1: Configuración del Entorno

La librería se configura mediante variables de entorno o un archivo `.env`. Para este caso de uso, crea un archivo `.env` en la raíz de tu proyecto con la siguiente configuración:

```dotenv
# .env

# Habilitar el generador de Ollama
OLLAMA_ENABLED=True
OLLAMA_MODEL="lfm2.5-thinking" # O el modelo que prefieras

# Configurar el modo de recuperación (sparse, dense, o hybrid)
# Para empezar, 'sparse' es el más sencillo ya que no requiere embeddings.
RETRIEVAL_MODE="sparse"

# Topología de persistencia:
# - local_split: SQLite (+ índice vectorial local en dense/hybrid)
# - elasticsearch: backend unificado; solo soporta dense/hybrid
PERSISTENCE_BACKEND="local_split"

# Ruta de la base de datos para almacenar los documentos
SQLITE_URL="sqlite:///./data/custom_app.db"

# Si usas PERSISTENCE_BACKEND=elasticsearch:
# ES_BASE_URL="http://localhost:9200"
# ES_DOCS_INDEX="rag-docs"
# ES_HISTORY_INDEX="rag-history"
# ES_SYSTEM_INDEX="rag-system"
# ES_TOMBSTONES_INDEX="rag-tombstones"

# Opcional: ajusta los parámetros de logging
LOG_LEVEL="INFO"
```

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

## Paso 3: Script de Ingesta de Datos

Necesitamos un script para orquestar el proceso de ingesta. Este script inicializará los componentes necesarios, usará el nuevo `Loader` personalizado y ejecutará el pipeline.

```python
# run_ingestion.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 1. Importar componentes de la librería
from local_rag_backend.settings import settings
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.infrastructure.persistence.sql import base as db_base

# 2. Importar Loader (custom)
from my_custom_loader import DictListLoader

# 3. Datos de ejemplo
my_data = [
    {"title": "Inteligencia Artificial", "content": "La IA es la simulación de procesos de inteligencia humana.", "metadata": {"category": "Tech"}},
    {"title": "Hexagonal Architecture", "content": "Es un patrón de diseño de software que desacopla el núcleo de la aplicación.", "metadata": {"category": "Software"}}
]

def main():
    print("--- Iniciando script de ingesta ---")

    # 4. Configurar la base de datos
    # Este ejemplo es para local_split/sparse. En elasticsearch no necesitas abrir SQLite.
    # Asegurarse de que el directorio de datos exista
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(settings.sqlite_url)
    db_base.ensure_sqlite_schema_compatible(engine_to_use=engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    # 5. Instanciar los componentes
    doc_storage = SqlDocumentStorage(session_factory=session_factory)
    custom_loader = DictListLoader(data=my_data)

    # El ETLService es necesario solo para modos 'dense' o 'hybrid'.
    # Para 'sparse', podemos interactuar directamente con el repositorio.
    # Ejemplo de cómo hacerlo de forma simple para 'sparse'.
    from local_rag_backend.core.services.ingestion import default_preprocess, default_chunker, default_formatter

    print(f"Cargando {len(my_data)} documentos...")
    all_chunks = []
    for item in custom_loader.load():
        clean_text = default_preprocess(item.text, item.metadata)
        chunks = default_chunker()(clean_text, item.metadata)
        for chunk in chunks:
            formatted_chunk = default_formatter(chunk, item.metadata)
            all_chunks.append(formatted_chunk)

    # 6. Almacenar los documentos procesados
    stored_ids = list(doc_storage.store_documents(all_chunks))
    print(f"\n[OK] Ingesta completada. {len(stored_ids)} chunks almacenados en la base de datos.")

if __name__ == "__main__":
    main()

```

Ejecuta el script para poblar tu base de datos:

```bash
python run_ingestion.py
```

Nota: este ejemplo escribe directo en SQLite para mantener el flujo simple (útil en `sparse`). Para el write-path canónico y consistente entre `sparse`/`dense`/`hybrid`, usa `MutationCoordinator` (sección de mutaciones más abajo).

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

## Mantenimiento (dense/hybrid): mutación canónica + repair explícito

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

# Repair explícito del estado de retrieval
rag-rebuild-index
```

### 2) API (FastAPI)

Levanta el servidor antes de llamar a la API:

```bash
rag-server
```

```bash
curl -X POST "http://localhost:8000/api/docs/mutate" \
  -H "Content-Type: application/json" \
  -d '{"op_id":"op-1","upserts":[{"external_id":"doc-1","content":"hola"}]}'

curl -X POST "http://localhost:8000/api/docs/mutate" \
  -H "Content-Type: application/json" \
  -d '{"op_id":"op-2","delete_external_ids":["chunk:<sha256>"]}'

curl -X POST "http://localhost:8000/api/index/rebuild"
```

Si has configurado `API_KEY`, añade `-H "X-API-Key: <API_KEY>"`. Además, por defecto las peticiones no-locales requieren API key.

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

---

## Operabilidad: métricas, evaluación y reranker

Monitoring mínimo (Prometheus):

```bash
uv sync --frozen --extra server --extra monitoring
export ENABLE_MONITORING=true
rag-server
curl -s http://localhost:8000/metrics | head
```

Si tienes `API_KEY`, añade `-H "X-API-Key: <API_KEY>"` al `curl`.

Evaluación offline reproducible (gate):

```bash
rag-eval --retrieval-mode sparse
```

Dataset por defecto: `datasets/rag_eval_v1.jsonl` (o `RAG_EVAL_DATASET_PATH`).

Reranker opcional (mejora de calidad medible con `rag-eval`):

```bash
export ENABLE_RERANKER=true
export RERANKER_CANDIDATE_K=20
```

# Guía de Uso Avanzado: Orquestación de RAG Local con Ollama

Este documento describe cómo utilizar `rag-prototype` como una librería de Python para construir flujos de trabajo de RAG (Retrieval-Augmented Generation) personalizados. Aprenderás a implementar tu propio cargador de datos (`Loader`) y a orquestar el proceso de ingesta y consulta utilizando un modelo local de Ollama.

## Requisitos Previos

1.  **Ollama en ejecución**: Asegúrate de tener Ollama instalado y un modelo descargado (ej. `ollama pull gemma3:4b`).
2.  **Proyecto instalado**: Instala el proyecto en modo editable para facilitar el desarrollo:

    ```bash
    uv venv .venv
    source .venv/bin/activate
    # Windows: .venv\Scripts\activate
    uv sync --frozen
    ```

---

## Paso 1: Configuración del Entorno

La librería se configura mediante variables de entorno o un archivo `.env`. Para este caso de uso, crea un archivo `.env` en la raíz de tu proyecto con la siguiente configuración:

```dotenv
# .env

# Habilitar el generador de Ollama
OLLAMA_ENABLED=True
OLLAMA_MODEL="gemma3:4b" # O el modelo que prefieras

# Configurar el modo de recuperación (sparse, dense, o hybrid)
# Para empezar, 'sparse' es el más sencillo ya que no requiere embeddings.
RETRIEVAL_MODE="sparse"

# Ruta de la base de datos para almacenar los documentos
SQLITE_URL="sqlite:///./data/custom_app.db"

# Opcional: ajusta los parámetros de logging
LOG_LEVEL="INFO"
```

## Paso 2: Implementación de un `Loader` Personalizado

La librería define una interfaz (`port`) para los cargadores de datos. Para crear el tuyo, solo necesitas implementar la clase `LoaderPort`.

Imagina que tus datos están en una lista de diccionarios. Así sería un `Loader` para ese formato:

```python
# my_custom_loader.py

from typing import Iterable, Any

from local_rag_backend.core.domain.entities import LoadedItem
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
            yield LoadedItem(text=text.strip(), metadata=metadata)

```

## Paso 3: Script de Ingesta de Datos

Ahora, crea un script para orquestar el proceso de ingesta. Este script inicializará los componentes necesarios, usará tu `Loader` personalizado y ejecutará el pipeline.

```python
# run_ingestion.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 1. Importar componentes de la librería
from local_rag_backend.settings import settings
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage
from local_rag_backend.infrastructure.persistence.sqlalchemy.base import Base

# 2. Importar tu Loader personalizado
from my_custom_loader import DictListLoader

# 3. Datos de ejemplo
my_data = [
    {"title": "Inteligencia Artificial", "content": "La IA es la simulación de procesos de inteligencia humana.", "metadata": {"category": "Tech"}},
    {"title": "Hexagonal Architecture", "content": "Es un patrón de diseño de software que desacopla el núcleo de la aplicación.", "metadata": {"category": "Software"}}
]

def main():
    print("--- Iniciando script de ingesta ---")

    # 4. Configurar la base de datos
    # Asegurarse de que el directorio de datos exista
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(settings.sqlite_url)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)

    # 5. Instanciar los componentes
    doc_storage = SqlDocumentStorage(session_factory=session_factory)
    custom_loader = DictListLoader(data=my_data)

    # El ETLService es necesario solo para modos 'dense' o 'hybrid'.
    # Para 'sparse', podemos interactuar directamente con el repositorio.
    # Aquí mostramos cómo hacerlo de forma simple para 'sparse'.
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

## Paso 4: Script de Consulta con Ollama

Finalmente, crea un script para hacer preguntas a tus datos utilizando el `RagService` y Ollama.

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

¡Y eso es todo! Siguiendo estos pasos, puedes usar este proyecto como una potente librería para construir sistemas RAG a medida, integrando tus propias fuentes de datos y aprovechando modelos locales con Ollama.

---

## CLI: ingesta desde ficheros/directorios (txt/md/csv)

Para un flujo rápido sin escribir código, puedes ingestar desde rutas locales:

```bash
# Ingesta desde un fichero o un directorio (recursivo por defecto)
rag-ingest ./docs ./notas.md ./data/faq.csv

# Ver qué se procesaría sin escribir en SQLite/FAISS
rag-ingest --dry-run ./docs
```

Notas:

* En `dense`/`hybrid`, la CLI actualiza SQLite y FAISS de forma consistente (y borra chunks obsoletos si un fichero se acorta).
* La detección de formato es best-effort (no solo extensión). Opcionalmente puedes instalar `python-magic` con el extra `magic`.

---

## Mantenimiento (dense/hybrid): borrado y rebuild idempotente del índice

En modos `dense`/`hybrid`, el índice FAISS es **estado derivado** de SQLite. Si borras filas manualmente en SQL o editas ficheros del índice a mano, puedes provocar **deriva** (IDs en FAISS que ya no existen en SQL, o documentos en SQL sin vector).

Opciones recomendadas:

### 1) CLI

```bash
# Borrar documentos por ID (SQL + FAISS cuando aplique)
rag-delete-docs 10 11 12

# Rebuild completo del índice desde SQLite (idempotente; dense/hybrid)
rag-rebuild-index
```

### 2) API (FastAPI)

```bash
# Borrar por IDs
curl -X POST "http://localhost:8000/api/docs/delete" \
  -H "Content-Type: application/json" \
  -d '{"ids":[10,11,12]}'

# Rebuild del índice (dense/hybrid)
curl -X POST "http://localhost:8000/api/index/rebuild"
```

Si has configurado `API_KEY`, añade `-H "X-API-Key: <API_KEY>"` a las llamadas.

### 3) Como librería (flujo programático)

Si orquestas tu propio pipeline y necesitas mantenimiento consistente entre stores:

```python
from local_rag_backend.core.services.maintenance import (
    delete_documents_multi_store,
    rebuild_index_from_db,
)
from local_rag_backend.infrastructure.persistence.sqlalchemy.sql_ import SqlDocumentStorage
from local_rag_backend.infrastructure.persistence.faiss.faiss_ import FaissVectorStorage
from local_rag_backend.infrastructure.embeddings.openai import OpenAIEmbedder

doc_repo = SqlDocumentStorage()
vec_repo = FaissVectorStorage(index_path="data/index.faiss", id_map_path="data/id_map.json", dim=None)
embedder = OpenAIEmbedder()

# 1) Borrado consistente
delete_documents_multi_store(doc_repo=doc_repo, vec_repo=vec_repo, embedder=embedder, ids=[10, 11, 12])

# 2) Rebuild idempotente desde SQLite
rebuild_index_from_db(doc_repo=doc_repo, vec_repo=vec_repo, embedder=embedder)
```

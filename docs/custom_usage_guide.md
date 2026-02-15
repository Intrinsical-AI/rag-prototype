# Guía de Uso Avanzado: Orquestación de RAG Local con Ollama

Este documento describe cómo utilizar `rag-prototype` como una librería de Python para construir flujos de trabajo de RAG (Retrieval-Augmented Generation) personalizados. Aprenderás a implementar tu propio cargador de datos (`Loader`) y a orquestar el proceso de ingesta y consulta utilizando un modelo local de Ollama.

## Requisitos Previos

1.  **Ollama en ejecución**: Asegúrate de tener Ollama instalado y un modelo descargado (ej. `ollama pull gemma3:1b`).
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
OLLAMA_MODEL="gemma3:1b" # O el modelo que prefieras

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

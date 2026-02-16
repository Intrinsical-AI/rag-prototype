# Curso: Ingeniería de Sistemas RAG en Producción (2026)

## Módulo 1: Fundamentos y Arquitectura de Software para IA
*El objetivo no es hacer un script, es construir un sistema mantenible.*

**Capítulo 1.1: Evolución del RAG (2023-2026)**
*   1.1.1. La tendencia "Local-First" y "Privacy-Preserving" (Ollama, Local Embeddings).
*   1.1.2. Análisis de cuellos de botella: Latencia vs. Precisión vs. Coste.

**Capítulo 1.2: Arquitectura Hexagonal (Ports & Adapters) en RAG**
*   1.2.1. El Hexágono: Desacoplando el Dominio (`core`) de la Infraestructura.
*   1.2.2. Definición de Protocolos y Puertos: `RetrieverPort`, `GeneratorPort`.
*   1.2.3. Inyección de Dependencias: `factory.py` y gestión de singletons en FastAPI.
*   1.2.4. Estrategias de Configuración: `pydantic-settings` y validación estricta.

---

## Módulo 2: Infraestructura, Hardware y Serving
*Basado en la sección "Serving / Application" del diagrama.*

**Capítulo 2.1: Gestión de Recursos de Cómputo**
*   2.1.1. Anatomía de un Request: ¿Qué consume CPU y qué consume GPU?
*   2.1.2. El rol de la CPU: I/O, Serialización JSON, Lógica de Negocio y Búsqueda Sparse (BM25).
*   2.1.3. El rol de la GPU: Aceleración de Embeddings y Generación / Inferencia LLM.
*   2.1.4. Gestión del GIL en Python: `ThreadPoolExecutor` y `asyncio` para no bloquear el Event Loop.

**Capítulo 2.2: Persistencia y Concurrencia "Bare Metal"**
*   2.2.1. El problema de la escritura concurrente en archivos locales (SQLite + FAISS).
*   2.2.2. Implementación de *File Locking* multiplataforma (`fcntl` vs `msvcrt`).
*   2.2.3. Escrituras Atómicas: Patrón *Write-to-Temp-and-Rename*.
*   2.2.4. Transacciones Distribuidas (SQL + Vector): Rollbacks de "mejor esfuerzo".

**Capítulo 2.3: Persistencia y Concurrencia Cloud**
*   2.2.1. El problema de la escritura concurrente en archivos locales (SQLite + FAISS).
*   2.2.2. Implementación de *File Locking* multiplataforma (`fcntl` vs `msvcrt`).
*   2.2.3. Escrituras Atómicas: Patrón *Write-to-Temp-and-Rename*.
*   2.2.4. Transacciones Distribuidas (SQL + Vector): Rollbacks de "mejor esfuerzo".

---

## Módulo 3: Pipelines de Ingesta e Integridad de Datos
*Basado en la sección "Ingestion" del diagrama y el código `ingestion.py` / `cli.py`.*

**Capítulo 3.1: ETL y Preprocesamiento**
*   3.1.1. Estrategias de Extracción: Adaptadores propios vs. Wrappers de LangChain.
*   3.1.2. Limpieza y Normalización Determinista.
*   3.1.3. Estrategias de Chunking: Ventanas deslizantes, semántico y recursivo.

**Capítulo 3.2: Idempotencia y Deduplicación**
*   3.2.1. El problema de la re-ingesta: Evitar vectores duplicados.
*   3.2.2. Hashing Semántico: `sha256(content + chunker_version + model)`.
*   3.2.3. Gestión de Identidad: `external_id` vs `internal_id`.

**Capítulo 3.3: Ciclo de Vida del Dato**
*   3.3.1. Actualizaciones (Upserts): Detección de cambios y regeneración de vectores.
*   3.3.2. Borrado Consistente y "Tombstones" (Soft Deletes).
*   3.3.3. "Self-Healing": Reconstrucción de índices corruptos desde la fuente de verdad (SQL).

---

## Módulo 4: Recuperación Avanzada (Retrieval)
*Basado en las secciones "Retrieval/Rerank" y "HNSW" del diagrama.*

**Capítulo 4.1: Algoritmos de Búsqueda**
*   4.1.1. Búsqueda Densa (Dense Retrieval): FAISS, HNSW y Flat Indexes.
*   4.1.2. Búsqueda Dispersa (Sparse Retrieval): BM25 y TF-IDF tokenización.
*   4.1.3. Limitaciones de los Embeddings: El problema de la coincidencia exacta de palabras clave.

**Capítulo 4.2: Búsqueda Híbrida (Hybrid Search)**
*   4.2.1. Fusión de Scores: Algoritmo RRF (Reciprocal Rank Fusion) vs. Weighted Sum.
*   4.2.2. Ajuste del hiperparámetro `alpha`: Balanceando semántica vs. palabras clave.
*   4.2.3. Implementación de `HybridRetriever` en código.

**Capítulo 4.3: Arquitecturas de Modelos de Recuperación**
*   4.3.1. Bi-Encoders: Velocidad para el primer paso de recuperación.
*   4.3.2. Cross-Encoders: Precisión para el reordenamiento (Reranking).
*   4.3.3. ColBERT (Late Interaction): Lo mejor de ambos mundos.

**Capítulo 4.4: Técnicas de optimización. Scalar-Quant vs Prod-Quant**

---

## Módulo 5: Mantenimiento y Operaciones (MLOps para RAG)

**Capítulo 5.1: Gestión de "Drift" en Índices Vectoriales**
*   5.1.1. El peligro de cambiar modelos de embedding en caliente.
*   5.1.2. Implementación de `index_manifest.json`: Control de versiones de configuración.
*   5.1.3. Validación al arranque: Detección de incompatibilidad dimensional.

**Capítulo 5.2: Evaluación Offline (RAGAS)**
*   5.2.1. Creación de "Golden Datasets" (Pregunta, Respuesta, Contexto Esperado).
*   5.2.2. Métricas de Recuperación: Recall@K, MRR (Mean Reciprocal Rank).
*   5.2.3. Métricas de Generación: Faithfulness (Fidelidad) y Answer Relevance.

**Capítulo 5.3: Monitorización en Producción**
*   5.3.1. Métricas Implícitas: Feedback de usuario, tiempo de lectura, "Copy to Clipboard".
*   5.3.2. Logging Estructurado y Trazabilidad.
*   5.3.3. Endpoints de Salud: `/health` (liveness) vs `/ready` (integridad de índices).
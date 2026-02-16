# Curso: Ingeniería de Sistemas RAG en Producción (2026)

## Módulo 1: Fundamentos y Arquitectura de Software para IA
*El objetivo no es hacer un script, es construir un sistema mantenible.*

**Capítulo 1.1: Evolución del RAG (2023-2026)**
*   1.1.1. La tendencia "Local-First" y "Privacy-Preserving" (Ollama, Local Embeddings).
*   1.1.2. Análisis de cuellos de botella: Latencia vs. Precisión vs. Coste.
*   1.1.3. SLMs (2B-7B) para tareas especializadas: limpieza, routing y reranking en edge.
*   1.1.4. Del "vector-only RAG" a arquitecturas híbridas (SQL + Vector + Graph + Web).

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
*   2.3.1. Object Storage para artefactos de índice (S3/Blob/GCS): versionado y consistencia.
*   2.3.2. Vector DB gestionada vs self-hosted (Pinecone/Qdrant/Weaviate): trade-offs operativos.
*   2.3.3. Latencia de red, locality y cachés en arquitecturas multi-región.
*   2.3.4. Control de concurrencia distribuido: leases, colas y jobs idempotentes.

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

**Capítulo 3.4: Enriquecimiento de Datos (Graph & Metadata)**
*   3.4.1. Extracción de entidades (NER), taxonomías y metadatos para filtrado semántico.
*   3.4.2. Introducción a GraphRAG: tripletas (Sujeto, Predicado, Objeto) como complemento al vector.
*   3.4.3. Construcción incremental de grafos desde texto no estructurado.
*   3.4.4. Estrategias híbridas de indexación: SQL + Vector + Grafo.

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

**Capítulo 4.3: Pre-Retrieval (Query Understanding & Transformation)**
*   4.3.1. Query Rewriting: normalización de intención y desambiguación.
*   4.3.2. Expansión de consultas: Multi-Query, RAG-Fusion y HyDE.
*   4.3.3. Query Routing: cuándo ir a SQL, Vector DB, Grafo o Web Search.
*   4.3.4. Decomposition: sub-querying para preguntas multi-hop o compuestas.

**Capítulo 4.4: Post-Retrieval (Context Engineering)**
*   4.4.1. Reranking con Cross-Encoders y criterios de relevancia contextual.
*   4.4.2. Context Window Management: presupuestos de tokens y truncado inteligente.
*   4.4.3. Context Compression: sumarización previa y filtrado por densidad informativa.
*   4.4.4. "Lost in the Middle": ordenamiento de chunks para maximizar atención del LLM.

**Capítulo 4.5: Arquitecturas de Modelos de Recuperación**
*   4.5.1. Bi-Encoders: velocidad para el primer paso de recuperación.
*   4.5.2. Cross-Encoders: precisión para el reordenamiento (Reranking).
*   4.5.3. ColBERT (Late Interaction): compromiso entre recall y precisión.

**Capítulo 4.6: Optimización de Índices y Coste**
*   4.6.1. Scalar Quantization vs Product Quantization.
*   4.6.2. HNSW tuning (`M`, `efConstruction`, `efSearch`) según perfil de carga.
*   4.6.3. Estrategias de actualización incremental vs rebuild completo.

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

**Capítulo 5.3: La "G" de RAG (Generación Controlada)**
*   5.3.1. Citations & Attribution: respuestas con fuentes trazables (`chunk_id`, `external_id`).
*   5.3.2. Structured Outputs: JSON mode, esquemas y validación con Pydantic.
*   5.3.3. Políticas de generación: groundedness, abstención y manejo de incertidumbre.

**Capítulo 5.4: Seguridad y Compliance**
*   5.4.1. Prompt Injection Indirecta: amenazas desde documentos ingeridos.
*   5.4.2. Sanitización de ingesta y de salida para evitar exfiltración y jailbreaks.
*   5.4.3. PII masking/redaction antes de enviar datos a LLMs de terceros.
*   5.4.4. Controles de acceso, auditoría y trazabilidad para entornos regulados.

**Capítulo 5.5: Monitorización en Producción**
*   5.5.1. Métricas Implícitas: Feedback de usuario, tiempo de lectura, "Copy to Clipboard".
*   5.5.2. Logging Estructurado y Trazabilidad.
*   5.5.3. Endpoints de Salud: `/health` (liveness) vs `/ready` (integridad de índices).


# 🏛️ Máster en Arquitectura y Operaciones RAG (Enterprise Edition 2026)

## Módulo 1: Fundamentos y Diseño de Sistemas de IA
*Contexto: No es solo buscar texto, es orquestar conocimiento y decisiones.*

**1.1. El Ecosistema RAG en 2026**
*   1.1.1. De pipelines lineales a Sistemas Agénticos y Grafos de Conocimiento.
*   1.1.2. El coste del "Alucinación Cero": Trade-offs entre latencia, coste y grounding.
*   1.1.3. Arquitectura de Referencia 2026: Hexagonal + Event-Driven + Agentic.

## Módulo 2: Ingeniería de Inferencia y Serving (High-Performance)
*Responde a: Serving moderno, vLLM, TGI, Capacity Planning.*

**2.1. Motores de Inferencia SOTA**
*   2.1.1. **vLLM a fondo:** PagedAttention, Continuous Batching y gestión de memoria KV-Cache.
*   2.1.2. **Optimización de Latencia:** Speculative Decoding y Chunked Prefill.
*   2.1.3. **TGI (Text Generation Inference):** Despliegue compatible con OpenAI API y métricas nativas.
*   2.1.4. Cuantización en producción (AWQ, GPTQ, EXL2): Impacto en recall vs. ahorro de VRAM.

**2.2. Arquitectura de Despliegue Escalable**
*   2.2.1. **KServe en Kubernetes:** Modelmesh, auto-scaling (HPA) basado en métricas de GPU y Scale-to-Zero.
*   2.2.2. **Estrategias de Caché Multi-nivel:**
    *   Caché de Embeddings (Exact match).
    *   Caché Semántica (GPTCache/Redis Vector): Invalidación por versión de corpus.
*   2.2.3. **Capacity Planning:** Cálculo de throughput (tokens/s), gestión de colas y *Backpressure*.
*   2.2.4. Quality of Service (QoS): Rate limiting y priorización por Tenant.

## Módulo 3: Ingesta Avanzada y Estructuras de Datos Complejas
*Responde a: GraphRAG, RAPTOR, ACLs granulares.*

**3.1. Más allá del Chunking Plano**
*   3.1.1. **RAPTOR:** Construcción de árboles recursivos de resúmenes para preguntas de alto nivel.
*   3.1.2. **GraphRAG (Construcción):** Extracción de entidades/relaciones con LLMs y detección de comunidades (Leiden algorithms).
*   3.1.3. **Deduplicación Robusta:** Hashing semántico vs. sintáctico y gestión de colisiones.

**3.2. Gestión de Identidad y Seguridad del Dato**
*   3.2.1. Diseño de **ACLs (Access Control Lists) a nivel de Chunk**: El problema del "Cross-Tenant Leakage".
*   3.2.2. Etiquetado de seguridad en vectores: Filtrado mandatorio en tiempo de búsqueda (Pre-computation/filtering).
*   3.2.3. "Soft Deletes" y Tombstones en índices distribuidos: Garantizar el derecho al olvido.

## Módulo 4: Retrieval SOTA y Arquitecturas de Decisión
*Responde a: HyDE, ColBERTv2, RRF, Agentic Routing.*

**4.1. Estrategias de Recuperación SOTA**
*   4.1.1. **GraphRAG (Querying):** Búsqueda Global (resúmenes de comunidad) vs. Local (vecinos de nodo).
*   4.1.2. **HyDE (Hypothetical Document Embeddings):** Ventajas en zero-shot y riesgos de alucinación inducida.
*   4.1.3. **ColBERTv2 (Late Interaction):** Arquitectura multi-vector para precisión de cross-encoder a velocidad de bi-encoder.
*   4.1.4. **Hybrid Fusion Avanzado:** Algoritmos RRF (Reciprocal Rank Fusion) para combinar Vector + Keyword + Graph.

**4.2. RAG Agéntico y Orquestación**
*   4.2.1. **Patrones de Ruteo:** Clasificación de queries (Simple vs. Complex vs. Ambiguous) para routing dinámico.
*   4.2.2. **Self-RAG y Corrective RAG (CRAG):** Bucles de reflexión y autoevaluación antes de responder.
*   4.2.3. **Implementación con LangGraph:** Gestión de estado, persistencia de hilos (checkpoints) y "Human-in-the-loop".
*   4.2.4. **Safety Switches:** Detección de bucles infinitos, límites de herramientas y presupuestos de tokens.

## Módulo 5: Ingeniería de Seguridad en LLMs (SecEng)
*Responde a: OWASP Top 10, Prompt Injection, Threat Modeling.*

**5.1. Threat Modeling para RAG**
*   5.1.1. **OWASP Top 10 for LLMs:** Auditoría práctica punto por punto.
*   5.1.2. Superficie de ataque: Exfiltración de datos vía Retrieval y ataques de canal lateral (Timing analysis).
*   5.1.3. Supply Chain Security: Verificación de modelos (safetensors, hash signing) y envenenamiento de datos (Data Poisoning).

**5.2. Defensas Activas y Arquitectura Segura**
*   5.2.1. **Prompt Injection:** Separación estricta de Datos vs. Instrucciones (ChatML/System roles), delimitadores XML y "Sandboxing".
*   5.2.2. **Guardrails de Entrada/Salida:** NeMo Guardrails o LlamaGuard para filtrado de contexto y sanidad de respuesta.
*   5.2.3. Prevención de SSRF (Server-Side Request Forgery) en agentes con herramientas/plugins.

## Módulo 6: Observabilidad End-to-End y Operaciones (LLMOps)
*Responde a: OpenTelemetry, SLI/SLO, Tracing real.*

**6.1. Observabilidad con Estándares Abiertos**
*   6.1.1. **OpenTelemetry (OTel) para GenAI:** Convenciones semánticas para Spans de LLM (`gen_ai.system`, `gen_ai.token_count`).
*   6.1.2. **Tracing Distribuido:** Traza completa: Ingesta → Vector DB → Reranker → LLM → Usuario.
*   6.1.3. Integración con herramientas nativas OTel (TruLens, Arize Phoenix o stacks ELK/Grafana).

**6.2. Definición de Calidad de Servicio (SLI/SLO)**
*   6.2.1. Métricas de Latencia: p95 y p99 de Time-to-First-Token (TTFT) y End-to-End.
*   6.2.2. Métricas de Calidad en Producción: Grounding Rate, Citation Precision y "I don't know" Rate.
*   6.2.3. Métricas de Negocio: Coste por Query, Tasa de Reescritura de usuario.

## Módulo 7: Gobierno, Riesgo y Cumplimiento (GRC)
*Responde a: EU AI Act, NIST AI RMF, ISO 42001.*

**7.1. Marcos de Gestión de Riesgos**
*   7.1.1. **NIST AI RMF 1.0:** Mapeo, Medición y Gestión de riesgos en sistemas RAG.
*   7.1.2. **ISO/IEC 42001:** Implementación de un Sistema de Gestión de IA certificable.

**7.2. Cumplimiento Normativo (Foco UE)**
*   7.2.1. **EU AI Act en la práctica:** Obligaciones para sistemas de propósito general (GPAI), transparencia y documentación técnica.
*   7.2.2. **Copyright y Data Governance:** Trazabilidad de fuentes, gestión de licencias en el corpus y políticas de retención de datos.
*   7.2.3. Logging de Auditoría: Qué registrar y durante cuánto tiempo para análisis forense.

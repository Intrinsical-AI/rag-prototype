

Aspecto	Detalle
Async no implementado	Clientes LLM síncronos (requests/OpenAI SDK), escalabilidad limitad
Faiss 	IndexFlatL2 no escalable a millones de vectores (sin IVF/HNSW

Área	Posibilidades
Performance	Async I/O, connection pooling, índices vectoriales avanzadas (IVF/HNSW, Hnswlib)
Retrieval avanzado	Parent-Doc Retrieval, Cross-Encoders, ColBERT, reranking automático
Formatos	PDFs nativo, emails con regex, HTML, Office docs (docx/xlsx)
Storage alternatives	PostgreSQL + pgvector, Elasticsearch, Weaviate, Pinecone
Observabilidad	OpenTelemetry full, distributed tracing, logs estructurados
Safety & RAG resilience	PII detection, prompt injection guards, jailbreak detection, confidence scoring
Escalabilidad	Distributed embeddings (Ray), multi-node FAISS, sharding
Análisis	GraphRAG, knowledge graphs, entity extraction
Evaluación online	Métricas A/B, feedback loops, user satisfaction tracking
Marketplace	Template library, pre-built adapters (Notion, Slack, GitHub)

Security	7/10	Guards proxy, API key, manifest integrity. -3 porque: no auth/authZ, sin rate limiting, sin PII detection, sin prompt injection guards
Performance	5.5/10	Sync todo. -4.5 porque: no async, no caching, FAISS es IndexFlatL2, sin connection pooling
Escalabilidad	4/10	Single-process/FAISS local. -6 porque: no distributed, no multi-node, no sharding, tokens limits obvios
UX/Developer Experience	7/10	CLI bueno, API clara. -3 porque: UI frontend minimal, curva aprendizaje config compleja (173 vars), docs duras
Comunidad/Adoption	3/10	Muy nicho (Intrinsical AI). -7 vs LangChain/LlamaIndex: sin ecosistema, sin marketplace, issues/discussions bajos
Producción Ready	6/10	Robusto a nivel técnico pero no llave-en-mano. -4 porque: falta observability avanzada, sin SLA guarantees, deployments manual-heavy
Innovación	6/10	Sólido pero no pionero. -4 porque: no GraphRAG, no safety features, no reranking automático, no knowledge graphs
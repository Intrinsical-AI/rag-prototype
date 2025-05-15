# Optional FAISS index generation for dense retrieval
if settings.retrieval_mode == "dense":
    embedding_dim = 384  # Fixed dimension for testing/demo purposes
    rng = np.random.default_rng(seed=42)
    vectors = rng.random((len(texts), embedding_dim), dtype=np.float32)
    _build_dense_index(vectors, ids)


OpenAIEmbedder	✔︎ unit tests controlan happy path, errores y dimensiones	Quizá añadir prueba de fallback DEFAULT_DIM si model desconocido.

RagService	    ✔︎ unit tests prueban flujo completo (retrieval + generation + persistencia)	Sugiero añadir assert sobre que save_qa_history realmente persiste en QaHistory.

API endpoints	✔︎ integration tests con TestClient (OpenAI y Ollama)	Muy sólido. Quizá incluir tests de validación de k (e.g. k=0 o fuera de rango)


Revisar uso de todos los settings: algunos (openai_max_tokens, openai_top_p) están definidos pero no usados.

Refactor: mover lógica de población de CSV a un helper testable (src/utils) aparte para simplificar init_rag_service.


- Probar Docker

- Probar Compose
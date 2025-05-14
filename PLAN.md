DONE
1. Datos & build	• Llena data/faq.csv (10 filas).



CURRENT:
• Ajusta build_index.py hasta que imprima “FAISS index written.”	python build_index.py sin errores.



NEXT:
2. Core verde	• Implementa lógica real de BM25Retriever, OpenAIGenerator.
• Opcional: history en db.models y crud.	pytest -q tests/unit debe salir “2 passed”.
3. End-to-end	• Con TestClient, asegúrate de recibir respuesta coherente.
• Añade assertions sobre JSON-schema.	pytest tests/integration verde.
4. Logging & errores	• Usa loguru o logging en adaptadores (level=INFO).
• Maneja exceptions de OpenAI con HTTPException(502, …).	Simula caída poniendo OPENAI_API_KEY="".
5. History	• Crea QaHistory model (question, answer, ts).
• Endpoint GET /api/history paginado (limit, offset).	curl /api/history?limit=3 devuelve lista.
6. UI mínima	• Sirve frontend/index.html con StaticFiles de FastAPI.
• Añade scroll auto para log.	Abre en navegador → flujo OK.
7. Docker & README	• docker build . < 700 MB.
• README tabla de variables, pasos, comandos.	Clona limpio en carpeta vacía y sigue README.
8. Bonus (si sobra tiempo)	• ModeratorAgent que marque contenido tóxico con openai.moderations.
• Switch OLLAMA=1 probado.	Ejecuta pytest con --ollama marker.
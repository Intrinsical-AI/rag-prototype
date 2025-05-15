# 🧠 Local-RAG Backend & Frontend

> **Goal**: Minimalist Retrieval-Augmented Generation Q&A prototype showcasing good design, implementation, testing, documentation, and scalability practices. Ready to RAG!

---

## 0. Table of Contents

1.  [Project Structure](#1-project-structure)
2.  [Why This Design?](#2-why-this-design)
3.  [Architecture at a Glance](#3-architecture-at-a-glance)
4.  [Quick Start](#4-quick-start)
5.  [Configuration](#5-configuration)
6.  [Running the Application](#6-running-the-application)
7.  [Running Tests](#7-running-tests)
8.  [API Endpoints](#8-api-endpoints)
9.  [Design Decisions & Trade-Offs](#9-design-decisions--trade-offs)
10. [Known Limitations](#10-known-limitations)
11. [Further Reading / Bonus](#11-further-reading--bonus)

---

## 1. Project Structure

A brief overview of the main directories:

*   `data/`: Contains the input CSV (`faq.csv`) and serves as the default location for the SQLite database and FAISS index.
*   `frontend/`: Contains the `index.html` file for the minimal user interface.
*   `scripts/`: Includes utility scripts, notably `build_index.py` for pre-populating the database and building vector indexes.
*   `src/`: The core application code.
    *   `adapters/`: Concrete implementations (drivers) for external services or specific algorithms (e.g., BM25, FAISS, OpenAI, Ollama).
    *   `app/`: FastAPI application setup, API routing, and dependency injection.
    *   `core/`: Domain logic, use cases (e.g., `RagService`), and abstract ports (interfaces). Pure Python.
    *   `db/`: SQLAlchemy models, CRUD operations, and database session management.
*   `tests/`: Automated tests.
    *   `unit/`: Unit tests for individual components, with subdirectories fatores/adapters`.
    *   `integration/`: Integration tests palavras-chave API endpoints.

---

## 2. Why This Design?

| Need (from task)        | Our Reasoning                                     | Implementation Choice                                     |
| :---------------------- | :----------------------------------------------- | :-------------------------------------------------------- |
| *Rapid Prototyping*     | Zero external infra; minimal pure-Python deps.   | SQLite + BM25 (default), FastAPI + Vanilla JS Frontend. |
| *Scalable Path*         | Ability to swap components without major refactor. | Ports & Adapters (Hexagonal Architecture).              |
| *AI Integration*        | Must work offline **or** with OpenAI.            | `GeneratorPort` → `OpenAIGenerator` / `OllamaGenerator`.  |
| *Data Handling*         | Basic knowledge base from CSV.                   | CSV ingested into SQLite; FAISS option for dense search.  |
| *Efficient Reviewer UX* | Clone → install → (build index) → test → run.  | `build_index.py` script, `.env` based `settings.py`.    |
| *Minimal UI*            | Simple, functional, no heavy frameworks.         | Single `index.html` with vanilla HTML/CSS/JS.             |

---

## 3. Architecture at a Glance

The application follows a Ports & Adapters (Hexagonal) architecture to promote separation of concerns and testability.

![Arch-Mermaid-Diagram](docs/arch-diagram.png)

**Dependency Rule:** Arrows of `import` statements primarily point inwards, towards the `src/core` components. This adheres to the Dependency Inversion Principle.

*   `(Optional)`  – A more detailed  [explanation](docs/architecture.md) of the Ports & Adapters architecture used.
---

## 4. Quick Start

1.  **Create Environment & Install Dependencies:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install .
    ```

2.  **(Optional) Create `.env` file:**
    Copy `.env.example` to `.env` (if provided) or create a new `.env` file in the project root. Configure your `OPENAI_API_KEY` if you plan to use OpenAI, or set `OLLAMA_ENABLED=true` if you have a local Ollama instance running with the required models.
    ```env
    # Example .env content
    # OPENAI_API_KEY="sk-yourkeyhere"
    # OLLAMA_ENABLED=true
    # OLLAMA_MODEL="deepseek-r1:1.5B"
    ```

3.  **Initialize Database & Build Index (Recommended First Step):**
    This script populates the SQLite database from `data/faq.csv` and builds vector indexes if dense retrieval is configured.
    ```bash
    python -m scripts.build_index
    ```
    *Note: The application will attempt to create DB tables on startup if they don't exist and populate from `data/faq.csv` if the `documents` table is empty (for sparse mode). However, running `build_index.py` is recommended, especially for dense mode.*

4.  **Run the Application:**
    ```bash
    uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    *   Wait some seconds, till the CLI logs output: "Application startup complete."
    *   Access the web UI at: `http://localhost:8000/`
    *   API (Swagger) docs at: `http://localhost:8000/docs`

**Docker (Alternative):**

1.  Build the Docker image:
    ```bash
    docker build -t local-rag-app .
    ```
2.  Run the Docker container:
    ```bash
    # 1) build & run sólo backend (OpenAI):
    docker build -t local-rag .
    docker run -p 8000:8000 -e OPENAI_API_KEY=<key> local-rag

    # 2) stack with Ollama:
    docker compose --profile with-ollama up --build

    ```

UI will be available on  http://localhost:8000/.


---

## 5. Configuration

The application is configured via environment variables (loaded from an `.env` file if present) managed by `src/settings.py`.

| Env Var                  | Default                        | Customize?     | Description                                              |
| ------------------------ | ------------------------------ | -------------- | -------------------------------------------------------- |
| `APP_HOST`               | `0.0.0.0`                      | 🔵 Default     | Host IP for FastAPI server.                              |
| `APP_PORT`               | `8000`                         | 🔵 Default     | Port for FastAPI server.                                 |
| `RETRIEVAL_MODE`         | `sparse` (`sparse` or `dense`) | 🔵 Default     | Retrieval mode (`sparse`: BM25, `dense`: FAISS vectors). |
| `SQLITE_URL`             | `sqlite:///./data/app.db`      | 🔵 Default     | SQLite database connection URL.                          |
| `FAQ_CSV`                | `data/faq.csv`                 | 🔵 Default     | Path to FAQ CSV file (used by `build_index.py`).         |
| `CSV_HAS_HEADER`         | `True`                         | 🔵 Default     | Indicates if CSV has a header row.                       |
| `INDEX_PATH`             | `data/index.faiss`             | 🔵 Default     | Path to FAISS index file (used in dense mode).           |
| `ID_MAP_PATH`            | `data/id_map.pkl`              | 🔵 Default     | Path to FAISS ID map file.                               |
| `OPENAI_API_KEY`         | `None`                         | 🟢 Recommended | Your OpenAI API key (required if using OpenAI).          |
| `OPENAI_MODEL`           | `gpt-3.5-turbo`                | 🔵 Default     | Model for OpenAI chat completions.                       |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small`       | 🔵 Default     | Model for OpenAI embeddings.                             |
| `OPENAI_TEMPERATURE`     | `0.2`                          | 🔵 Default     | Sampling temperature for OpenAI responses.               |
| `OPENAI_TOP_P`           | `1.0`                          | 🔵 Default     | Nucleus sampling (`top_p`) for OpenAI.                   |
| `OPENAI_MAX_TOKENS`      | `256`                          | 🔵 Default     | Max tokens for OpenAI responses.                         |
| `OLLAMA_ENABLED`         | `True`                         | ⚪ Optional     | Enable usage of local Ollama LLM server.                 |
| `OLLAMA_MODEL`           | `gemma3:4b`                    | ⚪ Optional     | Ollama model to use locally.                             |
| `OLLAMA_BASE_URL`        | `http://localhost:11434`       | 🔵 Default     | Base URL for local Ollama instance.                      |
| `OLLAMA_REQUEST_TIMEOUT` | `90`                           | 🔵 Default     | Request timeout in seconds for Ollama calls.             |

---

## 6. Running the Application
1.  **Set up your `.env` file** with necessary configurations (e.g., `OPENAI_API_KEY` or `OLLAMA_ENABLED=true`) and **ensure all dependencies are installed** (see Quick Start).
3.  **(Recommended) Run the `build_index.py` script** to populate the database and create vector indexes:
    ```bash
    python scripts/build_index.py
    ```
4.  **Start the FastAPI server:**
    ```bash
    uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
    ```
5.  **Access the application:**
    *   **Web Interface:** Open your browser to `http://localhost:8000/`
    *   **API Documentation (Swagger UI):** `http://localhost:8000/docs`

---

## 7. Running Tests

Ensure you are in the project root directory with your virtual environment activated.

```bash
make test
```

The test suite includes:
*   **Unit tests** for core logic, adapters (with mocked external services), and utility functions.
*   **Integration tests** for API endpoints, ensuring a configured in-memory SQLite database and mocked LLM calls.
The `conftest.py` file in the `tests/` directory handles the setup of the test environment, including the in-memory database and initial data loading for integration tests.

### 7.1. Test Coverage

Test coverage is measured using `pytest-cov`. To generate a report:

```bash
pytest --cov=src --cov-report=html # Open htmlcov/index.html to view the detailed HTML report.
# pytest --cov=src for a quick summary in the console:
```

File	statements	missing	excluded	branches	partial	coverage
src/utils.py	8	0	0	0	0	100%
src/settings.py	23	0	0	0	0	100%
src/db/models.py	12	0	0	0	0	100%
src/db/crud.py	16	0	0	2	0	100%
src/db/base.py	11	0	0	0	0	100%
src/core/rag.py	17	0	0	0	0	100%
src/core/ports.py	14	3	0	0	0	79%
src/app/main.py	45	12	0	4	1	69%
src/app/dependencies.py	96	6	0	28	9	88%
src/app/api_router.py	27	0	0	0	0	100%
src/adapters/retrieval/sparse_bm25.py	46	9	0	12	4	78%
src/adapters/retrieval/hybrid.py	25	1	0	6	1	94%
src/adapters/retrieval/dense_faiss.py	50	9	0	20	7	74%
src/adapters/generation/openai_chat.py	22	0	0	0	0	100%
src/adapters/generation/ollama_chat.py	33	2	0	2	0	94%
src/adapters/embeddings/sentence_transformers.py	9	0	0	0	0	100%
src/adapters/embeddings/openai.py	22	0	0	0	0	100%
Total	476	42	0	74	22	


## 8. API Endpoints

The primary API endpoints are exposed under the `/api` prefix.

| Method | Path           | Request Body (JSON)   | Response Body (JSON)                                 | Description                                     |
| :----- | :------------- | :-------------------- | :--------------------------------------------------- | :---------------------------------------------- |
| `GET`  | `/`            | N/A                   | HTML                                                 | Serves the frontend user interface.             |
| `POST` | `/api/ask`     | `{"question": "str", "k": "int"}` | `{"answer": "str", "source_ids": ["list_of_k_ints"]}` | Accepts (question,k), returns an AI-generated answer and source document IDs. |
| `GET`  | `/api/history` | Query Params: `limit`, `offset` | `[{"id": int, "question": "str", ...}]`        | Retrieves a list of past Q&A pairs.             |

Full interactive API documentation is available via Swagger UI at `/docs` when the server is running.

---

## 9. Design Decisions & Trade-Offs

| Aspect             | Chosen Approach                                       | Alternatives Considered        | Rationale                                                                          |
| :----------------- | :---------------------------------------------------- | :--------------------------- | :--------------------------------------------------------------------------------- |
| **Backend**        | Python, FastAPI                                       | Flask, Django                | FastAPI for performance, async support, Pydantic validation, auto API docs.        |
| **Frontend**       | Vanilla HTML, CSS, JavaScript                         | React, Vue, Jinja2 templates | Simplicity for rapid prototyping; fulfills minimal UI task requirement.            |
| **Architecture**   | Ports & Adapters (Hexagonal)                          | Monolithic, Layered          | Decoupling, testability, swappable components (e.g., different LLMs, DBs).       |
| **Data Store**     | SQLite                                                | In-memory list, CSV, Postgres| Minimal persistent storage with ORM (SQLAlchemy); simple setup.                  |
| **Data Ingestion** | CSV (`data/faq.csv`) processed by `scripts/build_index.py` | Direct DB input, API upload  | Simple for FAQ-like data; script allows pre-processing and index building.       |
| **Retrieval**      | BM25 (default, sparse), FAISS (optional, dense)       | TF-IDF, other vector DBs     | BM25 for zero-dependency quick start. FAISS to demonstrate embedding-based lookup. |
| **LLM Interface**  | `GeneratorPort` with `OpenAIGenerator`, `OllamaGenerator` | Direct SDK calls in service  | Abstraction allows easy switching between local (Ollama) and cloud (OpenAI) LLMs. |
| **Testing**        | Pytest, `TestClient`, `unittest.mock`                 | `unittest`                   | Pytest for cleaner syntax, powerful fixtures. `TestClient` for API integration.    |

- **FAISS index type**: `IndexFlatL2` (fastest to set up, recommended for prototypes and small corpora; even `IndexFlatIP` it's supposed to work well with normalized embeddings). Consider IVF or HNSW for larger-scale deployments).

---

## 10. Known Limitations
*   Dense retrieval only **ASSUMING  build_index.py have been already EXECUTED!!**
*   **Frontend Simplicity:** The UI is intentionally minimal. A production application would require a more robust frontend framework, state management, and enhanced UX.
*   **Error Handling:** While basic error handling is in place for API calls and LLM interactions, a production system would need more comprehensive and granular error management and reporting.
*   **Scalability of Default Retrieval:** BM25 is suitable for small datasets. For larger corpora, `dense` retrieval (FAISS) or a dedicated search engine (Elasticsearch) would be more appropriate. FAISS index building can be slow for very large datasets if not optimized.
*   **Security:** No authentication or rate-limiting is implemented on the API endpoints. Not suitable for public production deployment.
*   **`created_at` Timestamps:** Currently stored as timezone-aware datetimes in the DB (SQLite default behavior might vary). Pydantic serializes them to ISO 8601 strings for the API. UI might need timezone conversion if specific local times are required.
*   **FAISS Index Management:** The `build_index.py` script creates the FAISS index. For dynamic data, a more robust index update/management strategy would be needed.
*   **UI Testing:** Frontend functionality has been verified manually. Automated UI tests (e.g., Playwright, Selenium) are not included.

---

*(Made by IntrinsicalAI, Gemini 2.5-05-06, and GPT o3 (v14/05/2025))*
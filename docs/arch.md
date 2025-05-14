# Application Architecture: Ports & Adapters (Hexagonal)

## 1. Introduction

This document outlines the software architecture设计 for the Local-RAG Q&A application. The primary architectural pattern employed is **Ports & Adapters**, also known as **Hexagonal Architecture**. This choice was made to build a system that is decoupled, testable, and flexible, allowing for easier maintenance and evolution.

The core idea is to isolate the application's central business logic (the "domain") from external concerns such as a
the web framework (FastAPI), database interactions (SQLAlchemy), or specific AI model PIs (OpenAI, Ollama).

## 2. Core Concepts of Ports & Adapters

In a Hexagonal Architecture:

*   **The Hexagon (Application Core / Domain):**
    *   Contains the pure business logic, use cases, and domain entities.
    *   It is completely independent of any external technology or framework. In our project, this is primarily represented by `src/core/`.
    *   It defines **Ports** (interfaces) through which it interacts with the outside world.

*   **Ports (Interfaces):**
    *   Ports are essentially APIs defined by the application core. They dictate *what* the application can do or *what* it needs from the outside, without specifying *how*.
    *   There are two main types of ports:
        *   **Driving/Input Ports:** APIs exposed by the core for external actors (like a UI or API endpoint) to invoke application functionality (e.g., `RagService.ask()`). These are often directly represented by the public methods of core service classes.
        *   **Driven/Output Ports:** Interfaces defined by the core that describe services it *needs* from the outside world (e.g., a way to retrieve documents, a way to generate text). These are abstract interfaces (`Protocols` in Python) that external components will implement. Examples: `RetrieverPort`, `GeneratorPort`, `EmbedderPort` in `src/core/ports.py`.

*   **Adapters (Implementations):**
    *   Adapters sit "outside" the hexagon and implement the Ports, or use the Ports to drive the application. They handle the interaction with specific technologies.
    *   **Driving/Input Adapters:** These adapt incoming requests from a specific technology (e.g., an HTTP request from FastAPI) and call the corresponding input port on the application core. Our FastAPI request handlers in `src/app/api_router.py` act as driving adapters, translating HTTP requests into calls to `RagService`.
    *   **Driven/Output Adapters:** These implement the interfaces (driven ports) defined by the core. They provide concrete implementations for external services. Examples:
        *   `src/adapters/retrieval/SparseBM25Retriever.py` implements `RetrieverPort`.
        *   `src/adapters/generation/OpenAIGenerator.py` implements `GeneratorPort`.
        *   `src/adapters/embeddings/SentenceTransformerEmbedder.py` implements `EmbedderPort`.
        *   `src/db/crud.py` can be seen as part of a data persistence adapter, interacting with SQLAlchemy which in turn talks to SQLite.

## 3. Benefits in This Project

Adopting this architecture for the Local-RAG application provided several key benefits:

1.  **Testability:**
    *   The application core (`src/core/`) can be unit-tested in complete isolation, without needing a database, a web server, or actual LLM API calls. We can use dummy/mock implementations (Test Doubles) for the driven ports (e.g., `DummyRetriever`, `DummyGenerator` in `tests/unit/test_rag.py`).
    *   Adapters can also be tested independently, mocking the ports they interact with or testing their integration with the actual external services (though for external API calls, mocking is preferred in unit/integration tests).

2.  **Flexibility & Swappability:**
    *   **Multiple LLM Backends:** By defining a `GeneratorPort`, we were able to easily implement and switch between `OpenAIGenerator` (cloud-based) and `OllamaGenerator` (local) without changing the core `RagService` logic. Adding another LLM provider would simply mean creating a new adapter that implements `GeneratorPort`.
    *   **Multiple Retrieval Strategies:** The `RetrieverPort` allows for different retrieval methods (`SparseBM25Retriever`, `DenseFaissRetriever`, `HybridRetriever`) to be used interchangeably by the `RagService`.
    *   **Potential for Different Databases:** While currently using SQLAlchemy for SQLite, if we needed to switch to PostgreSQL or another database, the changes would be largely contained within the data persistence adapter (`src/db/`), minimizing impact on the core logic.

3.  **Decoupling from Frameworks:**
    *   The core application logic is not tied to FastAPI. If, in the future, we decided to expose the RAG functionality via a CLI, a different web framework, or a message queue, the `RagService` would remain unchanged. We would only need to write new driving adapters.

4.  **Clear Separation of Concerns:**
    *   The distinction between "what" the application does (core logic) and "how" it's delivered or "how" it interacts with external tools (adapters) makes the codebase easier to understand, navigate, and maintain.

## 4. Mapping Project Components to Hexagonal Architecture

*   **Application Core (Hexagon):**
    *   `src/core/rag.py` (`RagService`): Orchestrates the Q&A process (retrieve, generate). This is the primary use case.
    *   `src/core/ports.py`: Defines the `RetrieverPort`, `GeneratorPort`, and `EmbedderPort` interfaces that the `RagService` or other core components depend on.

*   **Driving Adapters (Input):**
    *   `src/app/api_router.py`: Contains FastAPI route handlers. These adapt incoming HTTP requests from the frontend (or API clients) and call methods on the `RagService` (obtained via dependency injection from `src/app/dependencies.py`).
    *   `frontend/index.html` (and its JavaScript): Acts as the user-facing entry point, driving the application through HTTP requests.

*   **Driven Adapters (Output):**
    *   **Retrieval Adapters (`src/adapters/retrieval/`)**:
        *   `SparseBM25Retriever`: Implements `RetrieverPort` using BM25.
        *   `DenseFaissRetriever`: Implements `RetrieverPort` using FAISS and an embedder.
        *   `HybridRetriever`: Implements `RetrieverPort` by combining other retrievers.
    *   **Generation Adapters (`src/adapters/generation/`)**:
        *   `OpenAIGenerator`: Implements `GeneratorPort` using the OpenAI API.
        *   `OllamaGenerator`: Implements `GeneratorPort` using a local Ollama instance.
    *   **Embedding Adapters (`src/adapters/embeddings/`)**:
        *   `SentenceTransformerEmbedder`: Implements `EmbedderPort` using `sentence-transformers`.
        *   `OpenAIEmbedder`: Implements `EmbedderPort` using OpenAI's embedding API.
    *   **Database Interaction (`src/db/`)**:
        *   `crud.py` functions (e.g., `get_documents`, `add_history`) act as part of a data persistence adapter. `RagService` uses these to fetch document content and save history. These functions use SQLAlchemy, which is an adapter to the SQLite database.

*   **Infrastructure / Configuration:**
    *   `src/settings.py`: Manages application configuration.
    *   `src.app.dependencies.py`: Handles the construction and injection of services (like `RagService` with its concrete retriever and generator adapters) into the FastAPI application. This is where the "wiring" of adapters to ports happens for the running application.
    *   `scripts/build_index.py`: An infrastructure script to prepare data and indexes, supporting the data layer.

## 5. Dependency Flow

The fundamental rule is that **dependencies flow inwards**: `Adapters` depend on `Ports` (interfaces defined in the Core), but the `Core` does not depend on any specific `Adapter`.

```mermaid
flowchart LR
    A[External World <br/> (FastAPI, Frontend, OpenAI API, Ollama, Database)]
    B(Adapters <br/> src/adapters/*, src/app/api_router.py, src.db/*)
    C(Ports <br/> src/core/ports.py)
    D[Application Core / Domain Logic <br/> src/core/rag.py]

    A -- Drives/Interacts with --> B
    B -- Implements / Calls --> C
    C -- Defined by --> D
    D -- Uses --> C

    subgraph "Dependency Rule: Outer depends on Inner"
        direction LR
        B --> C
        C --> D
    end
```

This inward dependency flow is key to the benefits mentioned above, especially testability and decoupling.

## 6. Conclusion

The Ports & Adapters architecture, while perhaps seeming like overkill for a "simple Q&A app," has provided a robust foundation for this project. It allowed for clear separation of concerns from the outset, facilitated isolated unit testing, and made it straightforward to support multiple retrieval and generation strategies (e.g., BM25/FAISS, OpenAI/Ollama). This structure positions the application well for future extensions or modifications with minimal disruption to the core business logic.

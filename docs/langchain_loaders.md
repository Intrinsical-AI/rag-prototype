# LangChain Loaders Adapter

This document explains how to ingest content from LangChain document loaders using the `LangChainLoader` adapter, which implements the project's `LoaderPort` interface.

## Installation

Install optional extras for loaders:

```bash
uv sync --frozen --extra loaders
# Or
# pip install rag-prototype[loaders]
```

## Quick start

```python
from langchain_community.document_loaders import WebBaseLoader
from local_rag_backend.core.services.etl import ETLService
from local_rag_backend.core.services.ingestion import IngestionPipeline
from local_rag_backend.infrastructure.ingestion.loaders import LangChainLoader

# 1) Prepare ETL (document store, vector store, embedder)
etl = ETLService(doc_repo, vector_repo, embedder)

# 2) Wrap any LangChain loader
lc_loader = WebBaseLoader(["https://example.com"])  # or DirectoryLoader, SitemapLoader, etc.
loader = LangChainLoader(lc_loader, drop_empty=True, metadata_filter={"lang": "en"})

# 3) Run the pipeline
pipeline = IngestionPipeline(loader=loader, etl_service=etl)
count = pipeline.run()
print(f"Ingested {count} chunks")
```

## Behavior and options

- `drop_empty=True` (default): skip documents with empty/whitespace-only content.
- `metadata_filter={...}`: only yield items whose metadata include all provided key/value pairs.
- The adapter expects each LangChain `Document` to have `page_content` and `metadata`. It gracefully falls back to dict-like objects (with `page_content`/`metadata` keys) or stringification when needed.

## Supported loaders

Any loader that returns LangChain `Document` objects (or dicts with the same fields) via `.load()` is supported, for example:

- `WebBaseLoader`
- `DirectoryLoader`
- `SitemapLoader`
- `UnstructuredFileLoader` (requires corresponding dependencies)

Refer to LangChain documentation for specific loader configuration.

## Troubleshooting

- If you see `ModuleNotFoundError: langchain_community`, ensure you installed the `loaders` extras.
- Some web loaders may require additional dependencies or network access; consider marking tests as `-m "not network"` in CI.
- When using large pages or PDFs, consider tuning the ingestion chunking parameters (`INGEST_CHUNK_CHARS`, `INGEST_CHUNK_OVERLAP`).

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
from local_rag_backend.composition.container import AppContainer
from local_rag_backend.core.use_cases.docs_mutation import (
    MutationCoordinator,
    MutationIntent,
    MutationUpsertInput,
)
from local_rag_backend.infrastructure.ingestion.loaders import LangChainLoader
from local_rag_backend.settings import settings

# 1) Wrap any LangChain loader
lc_loader = WebBaseLoader(["https://example.com"])  # or DirectoryLoader, SitemapLoader, etc.
loader = LangChainLoader(lc_loader, drop_empty=True, metadata_filter={"lang": "en"})

# 2) Convert LoaderPort items into a canonical mutation intent
upserts = []
for i, item in enumerate(loader.load()):
    locator = item.lineage.record_locator or f"item:{i}"
    upserts.append(
        MutationUpsertInput(
            external_id=f"{item.lineage.source_uri}#{locator}",
            content=item.text,
            source_id=item.lineage.source_uri,
            metadata=item.metadata,
        )
    )

# 3) Persist through the canonical write path
container = AppContainer.from_settings(settings)
coordinator = MutationCoordinator(settings_obj=settings, ports=container.docs_mutation_ports())
summary = coordinator.execute(
    MutationIntent(op_id="", upserts=tuple(upserts), source="langchain:web")
)
print(summary)
```

For application writes, keep `MutationCoordinator` as the final write path. Direct `ETLService`/`IngestionPipeline` examples bypass the mutation journal, write lock, and backend-specific consistency rules.

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
- When using large pages or PDFs, consider tuning the ingestion chunking parameters (`ingest_chunk_chars`, `ingest_chunk_overlap`).

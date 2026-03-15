"""HTTP transport schemas grouped by bounded context."""

from local_rag_backend.http.schemas.docs import (
    CanonicalImportDocItem,
    CanonicalImportRequest,
    CanonicalImportResponse,
    DocsMutateRequest,
    DocsMutateResponse,
    DocsQueryRequest,
    ImportResponse,
    IngestRequest,
    IngestResponse,
    UpsertDocItem,
    UpsertDocResult,
)
from local_rag_backend.http.schemas.index import RebuildIndexResponse
from local_rag_backend.http.schemas.meta import ConfigResponse, TemplateResponse
from local_rag_backend.http.schemas.openrouter import (
    OpenRouterGenerateRequest,
    OpenRouterGenerateResponse,
    OpenRouterUsage,
)
from local_rag_backend.http.schemas.rag_api_models import (
    AskEvalConfig,
    AskEvalRequest,
    AskEvalResponse,
    AskRequest,
    AskResponse,
    HistoryItem,
    QueryResult,
)
from local_rag_backend.http.schemas.shared import DocumentInDB

__all__ = [
    "AskEvalConfig",
    "AskEvalRequest",
    "AskEvalResponse",
    "AskRequest",
    "AskResponse",
    "CanonicalImportDocItem",
    "CanonicalImportRequest",
    "CanonicalImportResponse",
    "ConfigResponse",
    "DocsMutateRequest",
    "DocsMutateResponse",
    "DocsQueryRequest",
    "DocumentInDB",
    "HistoryItem",
    "ImportResponse",
    "IngestRequest",
    "IngestResponse",
    "OpenRouterGenerateRequest",
    "OpenRouterGenerateResponse",
    "OpenRouterUsage",
    "QueryResult",
    "RebuildIndexResponse",
    "TemplateResponse",
    "UpsertDocItem",
    "UpsertDocResult",
]

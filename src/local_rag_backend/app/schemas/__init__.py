"""HTTP transport schemas grouped by bounded context."""

from local_rag_backend.app.schemas.docs import (
    DeleteDocsByExternalIdRequest,
    DeleteDocsByExternalIdResponse,
    DeleteDocsRequest,
    DeleteDocsResponse,
    ImportResponse,
    IngestRequest,
    IngestResponse,
    UpsertDocItem,
    UpsertDocResult,
    UpsertDocsRequest,
    UpsertDocsResponse,
)
from local_rag_backend.app.schemas.index import RebuildIndexResponse
from local_rag_backend.app.schemas.meta import ConfigResponse, TemplateResponse
from local_rag_backend.app.schemas.openrouter import (
    OpenRouterGenerateRequest,
    OpenRouterGenerateResponse,
    OpenRouterUsage,
)
from local_rag_backend.app.schemas.rag import (
    AskEvalConfig,
    AskEvalRequest,
    AskEvalResponse,
    AskRequest,
    AskResponse,
    HistoryItem,
    QueryResult,
)
from local_rag_backend.app.schemas.shared import DocumentInDB

__all__ = [
    "AskEvalConfig",
    "AskEvalRequest",
    "AskEvalResponse",
    "AskRequest",
    "AskResponse",
    "ConfigResponse",
    "DeleteDocsByExternalIdRequest",
    "DeleteDocsByExternalIdResponse",
    "DeleteDocsRequest",
    "DeleteDocsResponse",
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
    "UpsertDocsRequest",
    "UpsertDocsResponse",
]

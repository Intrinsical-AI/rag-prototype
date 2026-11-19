"""Native private operations use the shared RAG owners and truthful MCP effects."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from local_rag_backend import mcp_server
from local_rag_backend.core.use_cases.results import MutationSummary


@pytest.mark.parametrize("payload", [{}, {"op_id": ""}, {"op_id": " "}, {"op_id": 12}])
def test_native_mutation_requires_explicit_identity_before_container(payload, monkeypatch):
    def forbidden():
        raise AssertionError("invalid mutation must not construct application state")

    monkeypatch.setattr(mcp_server, "get_container", forbidden)
    with pytest.raises(ValidationError):
        mcp_server.tool_mutate_docs(payload)


def test_native_mutation_routes_all_modes_through_shared_operation(monkeypatch):
    calls = []
    container = object()
    monkeypatch.setattr(mcp_server, "get_container", lambda: container)

    def mutate(**kwargs):
        calls.append(kwargs)
        return MutationSummary(op_id="explicit", inserted=1, deleted_sql=2, tombstoned=1)

    monkeypatch.setattr(mcp_server, "mutate_documents", mutate)
    payload = {
        "op_id": "explicit",
        "upserts": [{"external_id": "x", "content": "text"}],
        "delete_ids": ["id"],
        "delete_external_ids": ["tombstone"],
        "hard_delete_external_ids": ["hard"],
    }
    result = mcp_server.tool_mutate_docs(payload)
    assert result["op_id"] == "explicit" and result["deleted_sql"] == 2
    assert calls == [
        {
            "container": container,
            "payload": payload,
            "source": "mcp:rag_mutate_docs",
            "require_op_id": True,
        }
    ]


def test_native_embedding_tools_use_typed_api_and_preserve_model_identity(monkeypatch):
    from local_rag_backend.integrations import embeddings

    calls = []
    service = SimpleNamespace(
        status=lambda: {"provider": "local", "model": "configured", "dimension": 2},
        embed=lambda texts: calls.append(texts) or [[3.0, 4.0]],
    )
    monkeypatch.setattr(embeddings, "create_embedding_service", lambda: service)
    assert mcp_server.tool_embedding_status()["model"] == "configured"
    result = mcp_server.tool_embed_texts(["text"], normalize=True)
    assert result["vectors"] == [[0.6, 0.8]]
    assert calls == [["text"]]
    with pytest.raises(mcp_server.ToolOperationError, match="differs"):
        mcp_server.tool_embed_texts(["text"], model="other")
    assert calls == [["text"]]
    for name in ("rag_embedding_status", "rag_embed_texts"):
        assert not mcp_server.TOOLS[name].read_only
        assert mcp_server.TOOLS[name].open_world
    assert mcp_server.TOOLS["rag_search"].read_only
    assert mcp_server.TOOLS["rag_search"].open_world

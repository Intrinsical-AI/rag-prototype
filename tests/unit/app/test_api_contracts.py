"""
API contract validation tests for parameter bounds and validation.
"""

import pytest

from local_rag_backend.app import dependencies as deps
from local_rag_backend.app.main import app


class _MockRagService:
    """Mock RAG service with minimal methods for contract testing."""

    def ask(self, question: str, top_k: int = 3):
        """Mock ask method that returns a minimal valid response.

        Must match the real RagService.ask() return structure:
        {"answer": str, "docs": list[Document], "scores": list[float]}
        """
        return {
            "answer": "mock answer",
            "docs": [],
            "scores": [],
        }


@pytest.fixture(autouse=True)
def _mock_rag_service():
    """Override RAG service dependency for all tests in this module."""

    async def _override():
        return _MockRagService()

    app.dependency_overrides[deps.get_rag_service] = _override
    yield
    app.dependency_overrides.pop(deps.get_rag_service, None)


async def test_ask_rejects_k_out_of_range(asgi_client, in_memory_sqlite) -> None:
    """Test that /api/ask rejects k values outside valid range [1, 10]."""
    # k = 0 should be rejected
    response = await asgi_client.post("/api/ask", json={"question": "test", "k": 0})
    assert response.status_code == 422

    # k = 11 should be rejected
    response = await asgi_client.post("/api/ask", json={"question": "test", "k": 11})
    assert response.status_code == 422

    # k = 1 should be accepted (boundary)
    response = await asgi_client.post("/api/ask", json={"question": "test", "k": 1})
    assert response.status_code == 200

    # k = 10 should be accepted (boundary)
    response = await asgi_client.post("/api/ask", json={"question": "test", "k": 10})
    assert response.status_code == 200


async def test_ask_accepts_empty_question(asgi_client, in_memory_sqlite) -> None:
    """Test that /api/ask accepts empty questions (handled by business logic)."""
    # Empty string should be accepted by schema validation
    response = await asgi_client.post("/api/ask", json={"question": "", "k": 5})
    assert response.status_code == 200

    # Whitespace-only should be accepted by schema validation
    response = await asgi_client.post("/api/ask", json={"question": "   ", "k": 5})
    assert response.status_code == 200


async def test_history_limit_offset_bounds(asgi_client, in_memory_sqlite) -> None:
    """Test that /api/history validates limit and offset parameters."""
    # limit = 0 should be rejected
    response = await asgi_client.get("/api/history", params={"limit": 0})
    assert response.status_code == 422

    # limit = 101 should be rejected
    response = await asgi_client.get("/api/history", params={"limit": 101})
    assert response.status_code == 422

    # offset = -1 should be rejected
    response = await asgi_client.get("/api/history", params={"offset": -1})
    assert response.status_code == 422

    # Valid parameters should be accepted
    response = await asgi_client.get("/api/history", params={"limit": 10, "offset": 0})
    assert response.status_code == 200

    # Boundary values should be accepted
    response = await asgi_client.get("/api/history", params={"limit": 1, "offset": 0})
    assert response.status_code == 200

    response = await asgi_client.get("/api/history", params={"limit": 100, "offset": 0})
    assert response.status_code == 200


async def test_ask_missing_required_fields(asgi_client, in_memory_sqlite) -> None:
    """Test that /api/ask requires question field but k has default."""
    # Missing question field should be rejected
    response = await asgi_client.post("/api/ask", json={"k": 5})
    assert response.status_code == 422

    # Missing k field should be accepted (has default value of 3)
    response = await asgi_client.post("/api/ask", json={"question": "test"})
    assert response.status_code == 200

    # Missing both fields should be rejected (question is required)
    response = await asgi_client.post("/api/ask", json={})
    assert response.status_code == 422


async def test_ask_invalid_json(asgi_client, in_memory_sqlite) -> None:
    """Test that /api/ask handles invalid JSON gracefully."""
    # Invalid JSON should return 422
    response = await asgi_client.post(
        "/api/ask", content="invalid json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


async def test_history_invalid_parameter_types(asgi_client, in_memory_sqlite) -> None:
    """Test that /api/history validates parameter types."""
    # Non-integer limit
    response = await asgi_client.get("/api/history", params={"limit": "invalid"})
    assert response.status_code == 422

    # Non-integer offset
    response = await asgi_client.get("/api/history", params={"offset": "invalid"})
    assert response.status_code == 422

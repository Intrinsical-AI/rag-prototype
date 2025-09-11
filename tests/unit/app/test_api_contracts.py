"""
API contract validation tests for parameter bounds and validation.
"""

from fastapi.testclient import TestClient

from local_rag_backend.app.main import app


def test_ask_rejects_k_out_of_range(in_memory_sqlite) -> None:
    """Test that /api/ask rejects k values outside valid range [1, 10]."""
    client = TestClient(app)

    # k = 0 should be rejected
    response = client.post("/api/ask", json={"question": "test", "k": 0})
    assert response.status_code == 422

    # k = 11 should be rejected
    response = client.post("/api/ask", json={"question": "test", "k": 11})
    assert response.status_code == 422

    # k = 1 should be accepted (boundary)
    response = client.post("/api/ask", json={"question": "test", "k": 1})
    assert response.status_code == 200

    # k = 10 should be accepted (boundary)
    response = client.post("/api/ask", json={"question": "test", "k": 10})
    assert response.status_code == 200


def test_ask_accepts_empty_question(in_memory_sqlite) -> None:
    """Test that /api/ask accepts empty questions (handled by business logic)."""
    client = TestClient(app)

    # Empty string should be accepted by schema validation
    response = client.post("/api/ask", json={"question": "", "k": 5})
    assert response.status_code == 200

    # Whitespace-only should be accepted by schema validation
    response = client.post("/api/ask", json={"question": "   ", "k": 5})
    assert response.status_code == 200


def test_history_limit_offset_bounds(in_memory_sqlite) -> None:
    """Test that /api/history validates limit and offset parameters."""
    client = TestClient(app)

    # limit = 0 should be rejected
    response = client.get("/api/history", params={"limit": 0})
    assert response.status_code == 422

    # limit = 101 should be rejected
    response = client.get("/api/history", params={"limit": 101})
    assert response.status_code == 422

    # offset = -1 should be rejected
    response = client.get("/api/history", params={"offset": -1})
    assert response.status_code == 422

    # Valid parameters should be accepted
    response = client.get("/api/history", params={"limit": 10, "offset": 0})
    assert response.status_code == 200

    # Boundary values should be accepted
    response = client.get("/api/history", params={"limit": 1, "offset": 0})
    assert response.status_code == 200

    response = client.get("/api/history", params={"limit": 100, "offset": 0})
    assert response.status_code == 200


def test_ask_missing_required_fields(in_memory_sqlite) -> None:
    """Test that /api/ask requires question field but k has default."""
    client = TestClient(app)

    # Missing question field should be rejected
    response = client.post("/api/ask", json={"k": 5})
    assert response.status_code == 422

    # Missing k field should be accepted (has default value of 3)
    response = client.post("/api/ask", json={"question": "test"})
    assert response.status_code == 200

    # Missing both fields should be rejected (question is required)
    response = client.post("/api/ask", json={})
    assert response.status_code == 422


def test_ask_invalid_json(in_memory_sqlite) -> None:
    """Test that /api/ask handles invalid JSON gracefully."""
    client = TestClient(app)

    # Invalid JSON should return 422
    response = client.post(
        "/api/ask", data="invalid json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_history_invalid_parameter_types(in_memory_sqlite) -> None:
    """Test that /api/history validates parameter types."""
    client = TestClient(app)

    # Non-integer limit
    response = client.get("/api/history", params={"limit": "invalid"})
    assert response.status_code == 422

    # Non-integer offset
    response = client.get("/api/history", params={"offset": "invalid"})
    assert response.status_code == 422

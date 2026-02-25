# tests/unit/app/test_api_history.py
from local_rag_backend.infrastructure.persistence.sql import models


async def test_history_endpoint_returns_seeded_rows(asgi_client, in_memory_sqlite):
    SessionLocal = in_memory_sqlite

    # Seed history rows
    with SessionLocal() as session:
        session.add(models.QaHistory(question="Q1", answer="A1", source_ids=[1, 2]))
        session.add(models.QaHistory(question="Q2", answer="A2", source_ids=None))
        session.commit()

    resp = await asgi_client.get("/api/history", params={"limit": 10, "offset": 0})
    assert resp.status_code == 200
    data = resp.json()

    assert isinstance(data, list) and len(data) == 2

    got = {item["question"]: item for item in data}
    assert got["Q1"]["answer"] == "A1"
    assert got["Q1"]["source_ids"] == ["1", "2"]
    assert got["Q2"]["answer"] == "A2"
    assert got["Q2"]["source_ids"] == []

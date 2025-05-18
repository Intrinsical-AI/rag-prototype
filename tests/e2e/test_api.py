# tests/e2e/test_api.py
from fastapi.testclient import TestClient

from src.app.dependencies import get_rag_service
from src.app.main import app


class DummyRagSvc:
    def ask(self, question, top_k=3):
        return {
            "answer": f"eco:{question}",
            "docs": [],
            "scores": [],
        }


# ---------- override dependencia --------------------------------------------
app.dependency_overrides = {}
app.dependency_overrides[get_rag_service] = lambda: DummyRagSvc()

client = TestClient(app)


# ---------- tests -----------------------------------------------------------
def test_post_ask_endpoint():
    resp = client.post("/api/ask", json={"question": "hola", "k": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "eco:hola"
    assert data["sources"] == []


def test_get_root_frontend_not_found(tmp_path, monkeypatch):
    # Simulamos que no existe index.html
    monkeypatch.setattr("src.app.main.FRONTEND_DIR", tmp_path)
    resp = client.get("/")
    assert resp.status_code == 404

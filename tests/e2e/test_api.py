# tests/e2e/test_api.py
from fastapi.testclient import TestClient

from local_rag_backend.app.dependencies import get_rag_service
from local_rag_backend.app.main import app


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
    # Simulamos que no existe index.html ni en paquete ni en repo
    # 1) Forzar fallo al buscar recursos empaquetados
    monkeypatch.setattr(
        "local_rag_backend.app.main.resources.files",
        lambda *a, **k: object(),  # objeto sin .joinpath -> provocará excepción y fallback
    )
    # 2) Fallback del repo apunta a carpeta vacía
    monkeypatch.setattr("local_rag_backend.app.main.FRONTEND_DIR", tmp_path)
    resp = client.get("/")
    assert resp.status_code == 404

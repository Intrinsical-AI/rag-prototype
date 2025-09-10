# tests/unit/app/test_factory_singleton.py
from local_rag_backend.app import factory

def test_singleton_reset(monkeypatch):
    # finge dependencias rápidas
    monkeypatch.setattr(factory, "get_retriever", lambda: object())
    monkeypatch.setattr(factory, "get_generator", lambda: object())
    class H: 
        def save(self, *a, **k): pass
    monkeypatch.setattr("local_rag_backend.infrastructure.persistence.sqlalchemy.sql_.HistorySqlStorage", H)

    s1 = factory.get_rag_service(force_reload=True)
    s2 = factory.get_rag_service()
    assert s1 is s2
    factory.reset_rag_service()
    s3 = factory.get_rag_service()
    assert s3 is not s2

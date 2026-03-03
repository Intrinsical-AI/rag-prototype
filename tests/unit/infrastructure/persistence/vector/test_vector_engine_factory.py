import pytest

from local_rag_backend.infrastructure.persistence.vector import factory


class _DummyFaissEngine:
    backend = "faiss"


class _DummyNumpyEngine:
    backend = "numpy"


def test_build_vector_engine_auto_prefers_faiss_when_available(monkeypatch):
    monkeypatch.setattr(factory, "is_faiss_available", lambda: True, raising=True)
    monkeypatch.setattr(factory, "FaissEngine", _DummyFaissEngine, raising=True)
    monkeypatch.setattr(factory, "NumpyEngine", _DummyNumpyEngine, raising=True)

    engine = factory.build_vector_engine(backend="auto")
    assert isinstance(engine, _DummyFaissEngine)


def test_build_vector_engine_auto_uses_numpy_when_faiss_unavailable(monkeypatch):
    monkeypatch.setattr(factory, "is_faiss_available", lambda: False, raising=True)
    monkeypatch.setattr(factory, "FaissEngine", _DummyFaissEngine, raising=True)
    monkeypatch.setattr(factory, "NumpyEngine", _DummyNumpyEngine, raising=True)

    engine = factory.build_vector_engine(backend="auto")
    assert isinstance(engine, _DummyNumpyEngine)


def test_build_vector_engine_forced_numpy(monkeypatch):
    monkeypatch.setattr(factory, "NumpyEngine", _DummyNumpyEngine, raising=True)

    engine = factory.build_vector_engine(backend="numpy")
    assert isinstance(engine, _DummyNumpyEngine)


def test_build_vector_engine_forced_faiss_raises_when_backend_unavailable(monkeypatch):
    class _MissingFaiss:
        def __init__(self):
            raise RuntimeError("faiss not installed")

    monkeypatch.setattr(factory, "FaissEngine", _MissingFaiss, raising=True)

    with pytest.raises(RuntimeError, match="faiss not installed"):
        factory.build_vector_engine(backend="faiss")


def test_build_vector_engine_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported vector backend"):
        factory.build_vector_engine(backend="weird")

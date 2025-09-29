from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from local_rag_backend.core.services.ingestion import IngestionPipeline
from local_rag_backend.infrastructure.ingestion.loaders import LangChainLoader


class _DummyDoc:
    def __init__(self, content: str, metadata: Mapping[str, Any] | None = None) -> None:
        self.page_content = content
        self.metadata = dict(metadata) if metadata else {}


class _DummyLoaderList:

    def __init__(self, docs: list[Any]) -> None:
        self._docs = docs

    def load(self) -> Iterable[Any]:
        def _gen() -> Iterator[Any]:
            yield from self._docs

        return _gen()


class _DummyLoaderGenerator:
    def __init__(self, docs: list[Any]) -> None:
        self._docs = docs

    def load(self) -> Iterable[Any]:
        def _gen() -> Iterator[Any]:
            yield from self._docs

        return _gen()


def test_langchain_loader_basic_attribute_docs():
    docs = [
        _DummyDoc("Hello", {"source": "a"}),
        _DummyDoc("World", {"source": "b"}),
    ]
    loader = LangChainLoader(_DummyLoaderList(docs))
    items = list(loader.load())
    assert len(items) == 2
    assert items[0].text == "Hello"
    assert items[0].metadata == {"source": "a"}
    assert items[1].text == "World"
    assert items[1].metadata == {"source": "b"}


def test_langchain_loader_drops_empty_by_default():
    docs = [
        _DummyDoc("   ", {"id": 1}),
        _DummyDoc("non-empty", {"id": 2}),
    ]
    loader = LangChainLoader(_DummyLoaderList(docs))

    items = list(loader.load())
    assert [it.text for it in items] == ["non-empty"]


def test_langchain_loader_keep_empty_when_configured():
    docs = [
        _DummyDoc("   ", {"id": 1}),
        _DummyDoc("non-empty", {"id": 2}),
    ]
    loader = LangChainLoader(_DummyLoaderList(docs), drop_empty=False)

    items = list(loader.load())
    assert [it.text for it in items] == ["   ", "non-empty"]


def test_langchain_loader_metadata_filter_matches_all_keys():
    docs = [
        _DummyDoc("A", {"lang": "en", "type": "web"}),
        _DummyDoc("B", {"lang": "es", "type": "web"}),
        _DummyDoc("C", {"lang": "en", "type": "pdf"}),
    ]
    loader = LangChainLoader(
        _DummyLoaderList(docs), metadata_filter={"lang": "en", "type": "web"}
    )

    items = list(loader.load())
    assert [it.text for it in items] == ["A"]


def test_langchain_loader_dict_fallback():
    docs = [
        {"page_content": "X", "metadata": {"m": 1}},
        {"page_content": "Y", "metadata": {"m": 2}},
    ]
    loader = LangChainLoader(_DummyLoaderList(docs))
    items = list(loader.load())

    assert [it.text for it in items] == ["X", "Y"]
    assert [it.metadata for it in items] == [{"m": 1}, {"m": 2}]


def test_langchain_loader_generator_support():
    docs = [_DummyDoc(str(i)) for i in range(5)]
    loader = LangChainLoader(_DummyLoaderGenerator(docs))

    items = list(loader.load())
    assert [it.text for it in items] == ["0", "1", "2", "3", "4"]


def test_langchain_loader_stringify_fallback():
    class Weird:
        def __str__(self) -> str:  # no page_content / metadata
            return "weird-object"

    docs = [Weird()]
    loader = LangChainLoader(_DummyLoaderList(docs))

    items = list(loader.load())
    assert len(items) == 1
    assert items[0].text == "weird-object"
    assert items[0].metadata is None


def test_langchain_loader_works_with_ingestion_pipeline(monkeypatch):
    # Minimal ETL mocks
    class DummyDocRepo:
        def __init__(self) -> None:
            self.saved: list[str] = []

        def store_documents(self, texts: list[str]) -> list[int]:
            self.saved.extend(texts)
            return list(range(1, len(texts) + 1))

    class DummyEmbedder:
        dim = 3

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[float(i), 0.0, 0.0] for i, _ in enumerate(texts)]

    class DummyVectorRepo:
        def upsert(self, ids, vectors) -> None:
            pass

    from local_rag_backend.core.services.etl import ETLService

    etl = ETLService(DummyDocRepo(), DummyVectorRepo(), DummyEmbedder())

    docs = [
        _DummyDoc("Title A\n\nBody A", {"title": "Title A"}),
        _DummyDoc("Title B\n\nBody B", {"title": "Title B"}),
    ]
    loader = LangChainLoader(_DummyLoaderList(docs))

    pipeline = IngestionPipeline(loader=loader, etl_service=etl)
    count = pipeline.run()

    # Pipeline counts input chunks (no chunking splitting here because texts are short)
    assert count == 2

import pytest
from unittest.mock import Mock

from local_rag_backend.core.domain.entities import LoadedItem
from local_rag_backend.core.services.ingestion import (
    IngestionPipeline,
    default_chunker,
    default_formatter,
    default_preprocess,
)


class DummyLoader:
    def __init__(self, items):
        self.items = items

    def load(self):
        return iter(self.items)


class DummyETL:
    def __init__(self):
        self.ingested_texts = []

    def ingest(self, texts):
        self.ingested_texts.extend(texts)
        return list(range(len(self.ingested_texts) - len(texts) + 1, len(self.ingested_texts) + 1))


def test_default_preprocess():
    result = default_preprocess("  HELLO WORLD  ", {"title": "Test"})
    assert result == "hello world"


def test_default_chunker_no_split():
    chunker = default_chunker(max_chars=100, overlap=20)
    result = chunker("Short text")
    assert result == ["Short text"]


def test_default_chunker_with_split():
    chunker = default_chunker(max_chars=10, overlap=3)
    text = "This is a long text that needs chunking"
    result = chunker(text)
    
    assert len(result) > 1
    assert all(len(chunk) <= 10 for chunk in result)
    # Check overlap
    assert result[1].startswith(result[0][-3:])


def test_default_formatter_no_metadata():
    result = default_formatter("Content", None)
    assert result == "Content"


def test_default_formatter_with_metadata():
    metadata = {"title": "Test Title", "url": "http://example.com"}
    result = default_formatter("Content", metadata)
    expected = "Title: Test Title\nUrl: http://example.com\n\nContent"
    assert result == expected


def test_default_formatter_filters_none_values():
    metadata = {"title": "Test", "url": None, "date": "2023-01-01"}
    result = default_formatter("Content", metadata)
    expected = "Title: Test\nDate: 2023-01-01\n\nContent"
    assert result == expected


def test_ingestion_pipeline_single_item():
    items = [LoadedItem(text="Test content", metadata={"title": "Test"})]
    loader = DummyLoader(items)
    etl = DummyETL()
    
    pipeline = IngestionPipeline(loader, etl, batch_size=1)
    ids = pipeline.run()
    
    assert len(ids) == 1
    assert len(etl.ingested_texts) == 1
    assert "Title: Test" in etl.ingested_texts[0]
    assert "test content" in etl.ingested_texts[0]


def test_ingestion_pipeline_multiple_items():
    items = [
        LoadedItem(text="First content", metadata={"title": "First"}),
        LoadedItem(text="Second content", metadata={"title": "Second"}),
    ]
    loader = DummyLoader(items)
    etl = DummyETL()
    
    pipeline = IngestionPipeline(loader, etl, batch_size=2)
    ids = pipeline.run()
    
    assert len(ids) == 2
    assert len(etl.ingested_texts) == 2


def test_ingestion_pipeline_batching():
    items = [
        LoadedItem(text=f"Content {i}", metadata={"title": f"Title {i}"})
        for i in range(5)
    ]
    loader = DummyLoader(items)
    etl = DummyETL()
    
    pipeline = IngestionPipeline(loader, etl, batch_size=2)
    ids = pipeline.run()
    
    assert len(ids) == 5
    assert len(etl.ingested_texts) == 5


def test_ingestion_pipeline_chunking():
    # Long text that will be chunked
    long_text = "A" * 100
    items = [LoadedItem(text=long_text, metadata=None)]
    loader = DummyLoader(items)
    etl = DummyETL()
    
    chunker = default_chunker(max_chars=30, overlap=5)
    pipeline = IngestionPipeline(loader, etl, chunk=chunker)
    ids = pipeline.run()
    
    # Should create multiple chunks
    assert len(etl.ingested_texts) > 1
    assert all(len(text) <= 30 for text in etl.ingested_texts)


def test_ingestion_pipeline_custom_functions():
    items = [LoadedItem(text="test", metadata={"key": "value"})]
    loader = DummyLoader(items)
    etl = DummyETL()
    
    preprocess_mock = Mock(return_value="processed")
    chunk_mock = Mock(return_value=["chunk1", "chunk2"])
    format_mock = Mock(return_value="formatted")
    
    pipeline = IngestionPipeline(
        loader, etl,
        preprocess=preprocess_mock,
        chunk=chunk_mock,
        format_chunk=format_mock
    )
    ids = pipeline.run()
    
    preprocess_mock.assert_called_once_with("test", {"key": "value"})
    chunk_mock.assert_called_once_with("processed")
    assert format_mock.call_count == 2  # Called for each chunk
    assert len(etl.ingested_texts) == 2


def test_ingestion_pipeline_empty_loader():
    loader = DummyLoader([])
    etl = DummyETL()
    
    pipeline = IngestionPipeline(loader, etl)
    ids = pipeline.run()
    
    assert ids == []
    assert etl.ingested_texts == []

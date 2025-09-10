import tempfile
from pathlib import Path

import pytest

from local_rag_backend.core.domain.entities import LoadedItem
from local_rag_backend.infrastructure.ingestion.loaders.csv_loader import CSVLoader


def test_csv_loader_with_header_two_columns():
    csv_content = "Title;Content\nFirst Title;First content\nSecond Title;Second content"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(csv_content)
        temp_path = f.name
    
    try:
        loader = CSVLoader(temp_path, delimiter=";", has_header=True)
        items = list(loader.load())
        
        assert len(items) == 2
        assert items[0].text == "First Title\n\nFirst content"
        assert items[0].metadata == {"title": "First Title"}
        assert items[1].text == "Second Title\n\nSecond content"
        assert items[1].metadata == {"title": "Second Title"}
    finally:
        Path(temp_path).unlink()


def test_csv_loader_without_header():
    csv_content = "First Title;First content\nSecond Title;Second content"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(csv_content)
        temp_path = f.name
    
    try:
        loader = CSVLoader(temp_path, delimiter=";", has_header=False)
        items = list(loader.load())
        
        assert len(items) == 2
        assert items[0].text == "First Title\n\nFirst content"
        assert items[0].metadata == {"title": "First Title"}
    finally:
        Path(temp_path).unlink()


def test_csv_loader_single_column():
    csv_content = "Title\nJust content\nMore content"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(csv_content)
        temp_path = f.name
    
    try:
        loader = CSVLoader(temp_path, delimiter=";", has_header=True)
        items = list(loader.load())
        
        assert len(items) == 2
        assert items[0].text == "Just content"
        assert items[0].metadata is None
        assert items[1].text == "More content"
        assert items[1].metadata is None
    finally:
        Path(temp_path).unlink()


def test_csv_loader_empty_rows():
    csv_content = "Title;Content\nFirst;Content\n\nSecond;More content\n"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(csv_content)
        temp_path = f.name
    
    try:
        loader = CSVLoader(temp_path, delimiter=";", has_header=True)
        items = list(loader.load())
        
        # Empty rows should be skipped
        assert len(items) == 2
        assert items[0].text == "First\n\nContent"
        assert items[1].text == "Second\n\nMore content"
    finally:
        Path(temp_path).unlink()


def test_csv_loader_custom_delimiter():
    csv_content = "Title,Content\nFirst Title,First content\nSecond Title,Second content"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(csv_content)
        temp_path = f.name
    
    try:
        loader = CSVLoader(temp_path, delimiter=",", has_header=True)
        items = list(loader.load())
        
        assert len(items) == 2
        assert items[0].text == "First Title\n\nFirst content"
        assert items[0].metadata == {"title": "First Title"}
    finally:
        Path(temp_path).unlink()


def test_csv_loader_whitespace_handling():
    csv_content = "Title;Content\n  Spaced Title  ;  Spaced content  "
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(csv_content)
        temp_path = f.name
    
    try:
        loader = CSVLoader(temp_path, delimiter=";", has_header=True)
        items = list(loader.load())
        
        assert len(items) == 1
        assert items[0].text == "Spaced Title\n\nSpaced content"
        assert items[0].metadata == {"title": "Spaced Title"}
    finally:
        Path(temp_path).unlink()


def test_csv_loader_path_object():
    csv_content = "Title;Content\nTest;Content"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(csv_content)
        temp_path = Path(f.name)
    
    try:
        loader = CSVLoader(temp_path, delimiter=";", has_header=True)
        items = list(loader.load())
        
        assert len(items) == 1
        assert items[0].text == "Test\n\nContent"
    finally:
        temp_path.unlink()


def test_csv_loader_file_not_found():
    with pytest.raises(FileNotFoundError):
        loader = CSVLoader("nonexistent.csv")
        list(loader.load())

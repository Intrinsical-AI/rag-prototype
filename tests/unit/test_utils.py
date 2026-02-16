# tests/test_utils.py

from local_rag_backend.core.domain.entities import Document
from local_rag_backend.core.services.corpus import get_corpus_and_ids
from local_rag_backend.core.services.text_processing import preprocess_text


class DummyRepo:
    """Repo ligero para comprobar utilidades sin tocar BD."""

    def __init__(self):
        self._docs = [
            Document(id=1, content="Hello World"),
            Document(id=2, content="Second Doc"),
        ]

    def get_all_documents(self):
        return self._docs


def test_preprocess_text():
    raw = "  <b>Hello</b>\nWorld  "
    assert preprocess_text(raw) == "hello world"


def test_get_corpus_and_ids():
    repo = DummyRepo()
    corpus, ids = get_corpus_and_ids(repo)
    assert corpus == ["Hello World", "Second Doc"]
    assert ids == [1, 2]


def test_preprocess_inline_html():
    """Test that HTML tags are removed before collapsing spaces to prevent word concatenation."""
    # HTML tags between words should result in proper spacing
    assert preprocess_text("Hello<b></b>World") == "hello world"
    assert preprocess_text("Hello<span>Middle</span>World") == "hello middle world"

    # Multiple HTML tags
    assert preprocess_text("A<b>B</b><i>C</i>D") == "a b c d"

    # HTML with attributes
    assert preprocess_text('Hello<div class="test">World</div>End') == "hello world end"


def test_preprocess_html_with_whitespace():
    """Test HTML removal with existing whitespace."""
    # HTML tags with surrounding spaces
    assert preprocess_text("Hello <b>Bold</b> World") == "hello bold world"

    # Multiple spaces around HTML
    assert preprocess_text("Hello  <b>  Bold  </b>  World") == "hello bold world"


def test_preprocess_complex_html():
    """Test preprocessing with complex HTML structures."""
    html_text = """
    <div class="content">
        <p>First paragraph</p>
        <p>Second<br/>paragraph</p>
    </div>
    """
    result = preprocess_text(html_text)
    assert result == "first paragraph second paragraph"


def test_preprocess_edge_cases():
    """Test edge cases in text preprocessing."""
    # Empty string
    assert preprocess_text("") == ""

    # Only HTML tags
    assert preprocess_text("<div><p></p></div>") == ""

    # Only whitespace
    assert preprocess_text("   \n\t   ") == ""

    # Mixed case with HTML
    assert preprocess_text("HELLO<b>world</b>TEST") == "hello world test"


def test_preprocess_malformed_html():
    """Test preprocessing with malformed HTML."""
    # Unclosed tags
    assert preprocess_text("Hello<b>World") == "hello world"

    # Nested tags
    assert preprocess_text("Hello<b><i>World</i></b>End") == "hello world end"

    # Invalid tag syntax (should not be removed)
    assert preprocess_text("Hello<>World") == "hello<>world"

# tests/test_utils.py

from src.core.domain.entities import Document
from src.utils import get_corpus_and_ids, preprocess_text


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

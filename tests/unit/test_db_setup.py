# Ejemplo de test mínimo en tests/unit/test_db_setup.py (nuevo archivo)
from src.db.models import Document as DbDocument  # O el nombre que uses
from sqlalchemy.orm import Session


def test_basic_db_insert(db_session: Session):
    """Testea si se puede insertar en la tabla documents."""
    test_doc_id = 999
    test_doc_content = "basic insert test"
    doc = DbDocument(id=test_doc_id, content=test_doc_content)
    db_session.add(doc)
    db_session.commit()  # Esto fallará si la tabla no existe

    retrieved_doc = db_session.get(DbDocument, test_doc_id)
    assert retrieved_doc is not None
    assert retrieved_doc.content == test_doc_content

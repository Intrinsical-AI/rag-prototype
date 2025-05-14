# tests/unit/test_rag.py
"""
Unit-test del núcleo RagService sin depender de OpenAI ni BM25 reales,
y usando una base de datos en memoria para aislamiento.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session # Importar Session explícitamente

# Importar Base y modelos directamente para usarlos con el engine en memoria
from src.db.base import Base # El Base original para definir la estructura de tablas
from src.db.models import Document as DbDocument # Alias para evitar confusión con un posible Document de Pydantic

from src.core.rag import RagService
from src.core.ports import RetrieverPort, GeneratorPort # Importar los Puertos

# --- Test Doubles (Stubs) ---
class DummyRetriever(RetrieverPort): # Implementar el Puerto
    def __init__(self, doc_ids: list[int] = None, scores: list[float] = None):
        self.doc_ids_to_return = doc_ids if doc_ids is not None else [1]
        self.scores_to_return = scores if scores is not None else [0.9] * len(self.doc_ids_to_return)
        if len(self.doc_ids_to_return) != len(self.scores_to_return):
            raise ValueError("doc_ids y scores deben tener la misma longitud en DummyRetriever")

    def retrieve(self, query: str, k: int = 5) -> tuple[list[int], list[float]]:
        # Devolver solo hasta k resultados
        return self.doc_ids_to_return[:k], self.scores_to_return[:k]


class DummyGenerator(GeneratorPort): # Implementar el Puerto
    def __init__(self, answer: str = "dummy answer"):
        self.answer_to_return = answer

    def generate(self, question: str, contexts: list[str]) -> str:
        # Podrías incluso hacer asserts sobre 'question' o 'contexts' aquí si fuera necesario
        return self.answer_to_return


# --- Fixture para la base de datos en memoria ---
@pytest.fixture(scope="function") # 'function' scope para una BD limpia por test
def db_session() -> Session: # Tipar el retorno para claridad
    # Crear un engine SQLite en memoria para este test
    engine = create_engine("sqlite:///:memory:")
    # Crear todas las tablas definidas en Base.metadata (ej. Document, QaHistory)
    Base.metadata.create_all(bind=engine)

    # Crear una SessionLocal específica para este engine en memoria
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    db = TestingSessionLocal()
    try:
        yield db # Proporcionar la sesión al test
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine) # Limpiar después del test


# --- Tests ---
def test_rag_service_ask_retrieves_and_generates(db_session: Session):
    """
    Test que RagService.ask:
    1. Llama a retriever.retrieve().
    2. Llama a crud.get_documents() con los IDs del retriever.
    3. Llama a generator.generate() con la pregunta y los contextos.
    4. Devuelve la respuesta del generador y los IDs fuente.
    """
    # 1. Poblar la BD en memoria con un documento de prueba
    test_doc_id = 1
    test_doc_content = "dummy context content"
    doc = DbDocument(id=test_doc_id, content=test_doc_content)
    db_session.add(doc)
    db_session.commit()

    # 2. Configurar los Test Doubles
    retriever = DummyRetriever(doc_ids=[test_doc_id], scores=[0.95])
    generator = DummyGenerator(answer="specific dummy answer for this test")
    
    rag_service = RagService(retriever, generator)
    
    question = "hello?"

    # Act
    result = rag_service.ask(db_session, question)
    
    # Assert
    assert result["answer"] == "specific dummy answer for this test"
    assert result["source_ids"] == [test_doc_id]
    # Opcional: verificar que el historial se guardó (si QaHistory está definido en Base.metadata)
    # from src.db.models import QaHistory
    # history_entry = db_session.query(QaHistory).first()
    # assert history_entry is not None
    # assert history_entry.question == question
    # assert history_entry.answer == "specific dummy answer for this test"


def test_rag_service_ask_no_documents_found(db_session: Session):
    """
    Test que RagService.ask maneja el caso donde el retriever no devuelve IDs
    o los IDs no corresponden a documentos en la BD.
    El generador debería ser llamado con un contexto vacío.
    """
    # Arrange
    # BD está vacía (o los IDs devueltos no existen)
    
    retriever = DummyRetriever(doc_ids=[], scores=[]) # Retriever no devuelve nada
    
    # Queremos asegurar que el generador se llama con contextos vacíos
    class GeneratorSpy(GeneratorPort):
        called_with_contexts: list[str] | None = None
        def generate(self, question: str, contexts: list[str]) -> str:
            self.called_with_contexts = contexts
            return "generated with no context"

    generator_spy = GeneratorSpy()
    rag_service = RagService(retriever, generator_spy)
    question = "any question"

    # Act
    result = rag_service.ask(db_session, question)

    # Assert
    assert result["answer"] == "generated with no context"
    assert result["source_ids"] == []
    assert generator_spy.called_with_contexts == [] # Verifica que el contexto estaba vacío


def test_rag_service_uses_k_for_retrieval(db_session: Session):
    """
    Test que RagService.ask pasa el parámetro 'k' (o su default) al retriever.
    """
    # Arrange
    # Poblar la BD con suficientes documentos para que 'k' importe
    for i in range(1, 6):
        db_session.add(DbDocument(id=i, content=f"doc {i}"))
    db_session.commit()

    class RetrieverSpy(RetrieverPort):
        called_with_k: int | None = None
        def retrieve(self, query: str, k: int = 5) -> tuple[list[int], list[float]]:
            self.called_with_k = k
            # Devolver algunos IDs y scores válidos para que el flujo continúe
            actual_ids = [1, 2, 3, 4, 5]
            return actual_ids[:k], [0.9] * k 

    retriever_spy = RetrieverSpy()
    generator = DummyGenerator() # No nos importa la respuesta aquí
    rag_service = RagService(retriever_spy, generator)
    
    # Act
    rag_service.ask(db_session, "test question", k=3) # Llamar con k=3

    # Assert
    assert retriever_spy.called_with_k == 3

    # Act: Llamar sin k explícito (debería usar el default de RagService.ask, que es 3)
    rag_service.ask(db_session, "another test question")
    assert retriever_spy.called_with_k == 3 # El default de RagService.ask es k=3.
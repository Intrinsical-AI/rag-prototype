# src/infrastructure/persistence/models.py

from sqlalchemy import Column, DateTime, Integer, Text, func
from sqlalchemy.types import JSON  # <-- Nueva línea, si usas SQLite 3.9+

from src.infrastructure.persistence.sqlalchemy.base import Base


# ------------------------------------------------------------------ #
# Document ORM
# ------------------------------------------------------------------ #
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)


# ------------------------------------------------------------------ #
# BONUS: Q&A History
# ------------------------------------------------------------------ #
class QaHistory(Base):
    __tablename__ = "qa_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    source_ids = Column(
        JSON, nullable=True
    )  # <--- NUEVA COLUMNA (mejor JSON si SQLite lo soporta)

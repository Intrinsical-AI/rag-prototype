# ---------- stage 1: builder ----------
FROM python:3.11-slim AS builder

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV && \
    $VIRTUAL_ENV/bin/pip install --upgrade pip poetry

WORKDIR /app
COPY pyproject.toml README.md ./
RUN $VIRTUAL_ENV/bin/poetry install --only main --no-root

# ---------- stage 2: runtime ----------
FROM python:3.11-slim

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Copiamos entorno virtual ya resuelto por Poetry
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app

# Copiamos el código fuente (excluido por .dockerignore lo que no hace falta)
COPY src ./src
COPY frontend ./frontend
COPY data ./data       # semilla opcional

# Crear usuario no-root (buena práctica)
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Multi-stage Dockerfile for Intrinsical RAG Backend
# Reproducible builds via uv.lock + frozen sync
# =============================================================================

ARG PYTHON_IMAGE=python:3.11.11-slim-bookworm
ARG UV_VERSION=0.9.26

# --- Stage 1: Build base (includes build toolchain) ---
FROM ${PYTHON_IMAGE} as build-base
ARG UV_VERSION

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app
RUN python -m venv /opt/venv && pip install --no-cache-dir "uv==${UV_VERSION}"

# --- Stage 2: Dependencies (runtime deps + project) ---
FROM build-base as deps

ARG RAG_EXTRAS=""
COPY pyproject.toml uv.lock README.md LICENSE MANIFEST.in ./
COPY src/ ./src/
RUN set -eux; \
    EXTRA_FLAGS=""; \
    if [ -n "$RAG_EXTRAS" ]; then \
      for extra in $(echo "$RAG_EXTRAS" | tr ',' ' '); do \
        EXTRA_FLAGS="${EXTRA_FLAGS} --extra ${extra}"; \
      done; \
    fi; \
    uv sync --frozen --no-dev --extra server ${EXTRA_FLAGS}

# --- Stage 3: Development Environment ---
FROM deps as development

ARG RAG_EXTRAS=""
COPY . .
RUN set -eux; \
    EXTRA_FLAGS=""; \
    if [ -n "$RAG_EXTRAS" ]; then \
      for extra in $(echo "$RAG_EXTRAS" | tr ',' ' '); do \
        EXTRA_FLAGS="${EXTRA_FLAGS} --extra ${extra}"; \
      done; \
    fi; \
    uv sync --frozen --extra server --extra dev --extra test --extra lint ${EXTRA_FLAGS}

RUN mkdir -p data logs && chown -R appuser:appuser /app
USER appuser
CMD ["uvicorn", "local_rag_backend.http.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# --- Stage 4: Production runtime (minimal) ---
FROM ${PYTHON_IMAGE} as production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser appuser

COPY --from=deps /opt/venv /opt/venv

WORKDIR /app
RUN mkdir -p data logs && chown -R appuser:appuser /app

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=5)"

USER appuser
EXPOSE 8000
CMD ["uvicorn", "local_rag_backend.http.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--access-log", "--log-level", "info"]

# --- Stage 5: Testing Environment ---
FROM development as testing
CMD ["pytest", "-v"]

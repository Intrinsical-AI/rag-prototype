# =============================================================================
# Multi-stage Dockerfile for Intrinsical RAG Backend
# Optimized for production with development support
# =============================================================================

# --- Stage 1: Base Python Environment ---
FROM python:3.11-slim as python-base

# Python configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# --- Stage 2: Dependencies Installation ---
FROM python-base as deps

# Install uv for faster dependency management
RUN pip install uv

# Install runtime dependencies (and project) with uv
WORKDIR /tmp/app
COPY pyproject.toml ./
COPY README.md LICENSE MANIFEST.in ./
COPY src/ ./src/
ARG RAG_EXTRAS=""
RUN if [ -n "$RAG_EXTRAS" ]; then uv pip install --system ".[${RAG_EXTRAS}]"; else uv pip install --system .; fi

# --- Stage 3: Development Environment ---
FROM deps as development

# Set up development environment
WORKDIR /app
COPY . .

# Install development/test/lint extras in editable mode (project already present)
ARG RAG_EXTRAS=""
RUN if [ -n "$RAG_EXTRAS" ]; then uv pip install --system -e ".[${RAG_EXTRAS},dev,test,lint]"; else uv pip install --system -e ".[dev,test,lint]"; fi

# Create necessary directories
RUN mkdir -p data logs && \
    chown -R appuser:appuser /app

USER appuser

# Development server command
CMD ["uvicorn", "local_rag_backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# --- Stage 4: Production Environment ---
FROM python-base as production

# Copy installed dependencies (including the installed project) from deps stage.
# This avoids rebuilding the package (and potentially fetching build requirements) in this stage.
COPY --from=deps /usr/local /usr/local

# Set working directory
WORKDIR /app

# Create necessary directories and set permissions
RUN mkdir -p data logs && \
    chown -R appuser:appuser /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Production server command with optimizations (Uvicorn multi-workers)
CMD ["uvicorn", "local_rag_backend.app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--access-log", \
     "--log-level", "info"]

# --- Stage 5: Testing Environment ---
FROM development as testing

# Install test dependencies
RUN uv pip install --system --no-deps -e ".[test]"

# Run tests
CMD ["pytest", "-v"]

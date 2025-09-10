# Intrinsical RAG Prototype - Multi-stage Docker Build
# Production-ready containerization with security best practices

FROM python:3.11-slim AS builder

# Install system dependencies with security updates
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python dependencies with security and performance optimizations
WORKDIR /app

# Copy dependency files first for better Docker layer caching
COPY pyproject.toml README.md MANIFEST.in ./
COPY requirements.txt* ./

# Install dependencies with security flags
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --compile --no-deps uv && \
    pip install --no-cache-dir .

# Copy source code after dependencies (better cache invalidation)
COPY src ./src
COPY frontend ./frontend
COPY data ./data
COPY scripts ./scripts

# ---------- Production stage ----------
FROM python:3.11-slim AS production

# Install minimal runtime dependencies with security updates
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    ca-certificates \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/*

# Copy virtual environment from builder with proper permissions
COPY --from=builder /opt/venv /opt/venv

# Set Python environment variables for production
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app directory and non-root user with minimal privileges
WORKDIR /app
RUN groupadd -r appgroup --gid 1000 && \
    useradd -r -g appgroup --uid 1000 --home-dir /app --shell /bin/bash appuser

# Copy application code with proper ownership
COPY --from=builder --chown=appuser:appgroup /app/src ./src
COPY --from=builder --chown=appuser:appgroup /app/frontend ./frontend
COPY --from=builder --chown=appuser:appgroup /app/data ./data
COPY --from=builder --chown=appuser:appgroup /app/scripts ./scripts

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/.cache /app/logs && \
    chown -R appuser:appgroup /app && \
    chmod -R 755 /app

# Switch to non-root user
USER appuser

# Health check with proper endpoint and timing
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Expose port
EXPOSE 8000

# Security labels and metadata
LABEL maintainer="Intrinsical AI <contact@intrinsical.ai>" \
      version="1.0.0" \
      description="Intrinsical RAG Prototype - Production Container" \
      org.opencontainers.image.source="https://github.com/intrinsical-ai/rag-prototype" \
      org.opencontainers.image.vendor="Intrinsical AI" \
      org.opencontainers.image.licenses="MIT"

# Default command with proper signal handling
CMD ["rag-server", "--host", "0.0.0.0", "--port", "8000"]

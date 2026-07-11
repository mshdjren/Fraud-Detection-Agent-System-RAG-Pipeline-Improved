# Router Service Dockerfile
# Multi-stage build for GKE Elasticsearch environment

# Stage 1: Builder (빌드 도구로 의존성 컴파일)
FROM python:3.12-slim-bookworm as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements
# COPY requirements.txt .
# COPY common/requirements-base.txt ./requirements-base.txt

COPY requirements.txt requirements.txt

# Install to /install directory (System Python 사용)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime (빌드 도구 제거, 최소 이미지)
FROM python:3.12-slim-bookworm

# Copy compiled packages from builder
COPY --from=builder /install /usr/local

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser orchestrator.py .

USER appuser

EXPOSE 8080
# Run application
CMD ["python", "-m", "uvicorn", "orchestrator:app", "--host", "0.0.0.0", "--port", "8080"]

# ===========================
# Streamlit UI Dockerfile
# ===========================
# PyTorch 없음 → 기존 analyzer Dockerfile 구조 유지하면서 경량화

FROM python:3.12-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime ───────────────────────────────────────────────
FROM python:3.12-slim-bookworm

COPY --from=builder /install /usr/local

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

COPY --chown=appuser:appuser streamlit.py .
# COPY --chown=appuser:appuser streamlit.yaml .

# ENV CONFIG_PATH=/app/streamlit.yaml

USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "streamlit.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
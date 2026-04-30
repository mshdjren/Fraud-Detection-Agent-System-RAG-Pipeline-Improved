# Stage 1: Builder (기존과 동일하게 gcc/g++ 설치되어 있으므로 유지)
FROM python:3.12-slim-bookworm as builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim-bookworm

# ★ 핵심: FAISS 실행을 위한 런타임 라이브러리(libomp5) 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libomp5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app
COPY --chown=appuser:appuser retriever.py .

USER appuser
EXPOSE 8001

CMD ["python", "-m", "uvicorn", "retriever:app", "--host", "0.0.0.0", "--port", "8001"]
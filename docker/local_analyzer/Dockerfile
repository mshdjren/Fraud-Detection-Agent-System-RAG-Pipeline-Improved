# ===========================
# Local Analyzer Dockerfile
# ===========================
# 기존 analyzer Dockerfile의 multi-stage 구조 그대로 유지.
# 차이점:
#   - PyTorch CPU-only wheel (--index-url 별도 지정)
#   - Vertex AI / LangChain 없음
#   - 포트 8080 (기존 analyzer: 8002)
#   - HuggingFace base model은 런타임 다운로드 (이미지에 포함 X)
#   - LoRA adapter는 init container가 GCS에서 /data/adapter 에 다운로드

# ── Stage 1: Builder ──────────────────────────────────────
FROM python:3.12-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# PyTorch CPU wheel은 공식 index-url 필요 (PyPI에는 CUDA 빌드만 있음)
# 나머지 패키지는 PyPI에서 정상 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install \
        torch==2.5.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --prefix=/install \
        -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

# ── Stage 2: Runtime ──────────────────────────────────────
FROM python:3.12-slim-bookworm

# 기존 analyzer Dockerfile과 동일한 사용자/디렉터리 구조
COPY --from=builder /install /usr/local

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /data && \
    chown -R appuser:appuser /app /data

WORKDIR /app

# 애플리케이션 코드
# config.yaml은 ConfigMap volumeMount로 /app에 주입되므로
# 이미지에는 포함하지 않아도 되지만 로컬 테스트용으로 COPY
COPY --chown=appuser:appuser local_analyzer.py .
# COPY --chown=appuser:appuser config.yaml .

# HuggingFace 캐시 위치를 /data 아래로 고정
# → init container가 마운트하는 emptyDir(/data)와 동일 볼륨
# → base model도 Pod 재시작 시 재다운로드 (캐시 없음, 허용 범위)
ENV HF_HOME=/data/hf_cache
ENV TRANSFORMERS_CACHE=/data/hf_cache

# CONFIG_PATH 기본값 (yaml volumeMount가 없을 때 fallback)
# ENV CONFIG_PATH=/app/config.yaml

USER appuser

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "local_analyzer:app", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--timeout-keep-alive", "300"]
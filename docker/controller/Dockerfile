# Controller Service Dockerfile (Planner + LangGraph Agent Loop)
# Multi-stage build for GKE Elasticsearch environment
#
# 기존 06-controller.yaml은 anomaly-analyzer:1.0 이미지를 재사용했으나,
# controller.py가 LangGraph StateGraph로 agent loop 전체(retrieve→analyze
# →plan→tool→loop)를 호스팅하게 되면서 langgraph 의존성이 추가로 필요해져
# 더 이상 analyzer 이미지로는 동작하지 않는다. 이 Dockerfile로 controller
# 전용 이미지를 별도 빌드한다.

# Stage 1: Builder (빌드 도구로 의존성 컴파일)
FROM python:3.12-slim-bookworm as builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements-controller.txt requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime (빌드 도구 제거, 최소 이미지)
FROM python:3.12-slim-bookworm

COPY --from=builder /install /usr/local

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

# 코드는 06-controller.yaml의 code-volume(ConfigMap)이 런타임에 마운트하므로
# 이미지 안에는 포함하지 않는다 — 기존 controller/orchestrator 배포 패턴과
# 동일(volumeMounts로 controller.py/agent_tools.py/analyzer_schema.py 주입).
# 로컬에서 ConfigMap 없이 단독 실행/테스트할 경우를 위해 COPY는 유지한다.
COPY --chown=appuser:appuser controller.py .
COPY --chown=appuser:appuser agent_tools.py .
COPY --chown=appuser:appuser analyzer_schema.py .

USER appuser

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "controller:app", "--host", "0.0.0.0", "--port", "8080"]

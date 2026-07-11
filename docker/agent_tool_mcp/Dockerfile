# Agent Tools MCP Server Dockerfile
# Multi-stage build for GKE Elasticsearch environment
#
# expand_within_cluster / reroute_cross_cluster 를 표준 MCP tool로 노출하는
# 신규 서비스. controller.py의 TOOL_EXECUTION_MODE=mcp 비교실험을 위해
# Router/Retriever와 동급의 독립 마이크로서비스로 배포한다.

FROM python:3.12-slim-bookworm as builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements-agent-tools-mcp.txt requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim-bookworm

COPY --from=builder /install /usr/local

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

# code-volume(ConfigMap)이 런타임에 agent_tools.py/agent_tools_mcp_server.py를
# 마운트하므로 이미지 안에는 포함하지 않는다. 로컬 단독 실행/테스트용으로
# COPY는 유지한다.
COPY --chown=appuser:appuser agent_tools.py .
COPY --chown=appuser:appuser agent_tools_mcp_server.py .

USER appuser

EXPOSE 8080
# streamable-http transport — REST shim(/tools/*)과 표준 MCP(/mcp)를
# 동일 ASGI app으로 함께 서빙한다 (agent_tools_mcp_server.py의 main() 참고).
CMD ["python", "agent_tools_mcp_server.py", "--transport", "http", "--host", "0.0.0.0", "--port", "8080"]

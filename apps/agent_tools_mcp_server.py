"""
agent_tools_mcp_server.py — Follow-up Tools as a standard MCP server
======================================================================
목적:
  agent_tools.py에 정의된 follow-up tool 카탈로그(expand_within_cluster,
  reroute_cross_cluster, [compare_features])를, 코드로 직접 함수 호출하는
  대신 "Model Context Protocol" 표준 tool로 노출한다.

  실무에서 빅테크가 쓰는 패턴 그대로:
    - 공식 `mcp` Python SDK(FastMCP)로 tool을 정의 — @mcp.tool() 데코레이터가
      함수 시그니처(type hints + docstring)로부터 JSON Schema를 자동 생성.
      → controller.py(planner)가 LLM에게 보여주는 tool 카탈로그
        (agent_tools.build_followup_tool_catalog) 와 별개로, MCP client가
        표준 `tools/list`로 동일 정보를 받을 수 있게 된다.
    - Transport 이중화: stdio(로컬 프로세스 간, Claude Desktop류) +
      streamable-http(원격 마이크로서비스, GKE) 양쪽을 동일 FastMCP 인스턴스
      에서 제공 — "tool을 app으로" 배포하는 패턴.
    - retrieval 로직(Elasticsearch HNSW 호출)을 tool 내부에 캡슐화해,
      LLM-agnostic하게 만든다 (Gemini든 다른 LLM이든 동일 tool을 호출).

  기존 agent_tools.py(순수 카탈로그/메타데이터 모듈, controller.py의 LLM
  prompt 생성용 단일 출처)는 변경하지 않는다. 이 파일은 그 카탈로그의
  "실행체"만 MCP tool로 승격하는 별도 레이어다.

배포 형태:
  - 독립 FastAPI/MCP 마이크로서비스로 GKE에 배포 (Router/Retriever와 동급).
  - controller.py는 이 서비스를 두 가지 경로로 호출 가능:
      (a) 표준 MCP client (mcp.client.streamable_http) — 진짜 MCP 프로토콜
      (b) REST shim `/tools/{name}/call` — 기존 controller.py가 이미
          httpx로 호출하도록 짜여 있어 마이그레이션 비용 0
  두 경로 모두 동일한 tool 함수(_expand_within_cluster_impl 등)를 호출하므로
  동작은 동일하고, transport만 다르다 — pasted 문서의 "3-axis transport
  benchmark"(Direct call vs stdio vs SSE/HTTP, MCP 오버헤드 ~12.5ms 측정)를
  그대로 이 서비스에서 수행할 수 있다.

Run:
  # streamable-http (GKE에 띄울 기본 모드)
  python agent_tools_mcp_server.py --transport http --port 8090

  # stdio (로컬 client에 직결 — 디버깅/CLI client용)
  python agent_tools_mcp_server.py --transport stdio
"""

from __future__ import annotations

import os
import argparse
from typing import Any, Dict, List, Optional

import json as _json

import httpx
import structlog
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
from starlette.requests import Request
from starlette.responses import JSONResponse

# agent_tools.py — 카탈로그/메타데이터 단일 출처. 이 서비스는 그 메타데이터의
# "실행"만 담당하므로, use_when/returns 문구를 다시 정의하지 않고 그대로 재사용.
from agent_tools import (
    FOLLOWUP_TOOLS,
    FOLLOWUP_ACTIONS,
    ENABLE_COMPARE_FEATURES,
    compute_feature_comparison,
)

logger = structlog.get_logger()

# ============================================================
# Configuration — Retriever/Elasticsearch 호출 대상
# ============================================================
RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://anomaly-retriever:80")
ROUTER_URL    = os.getenv("ROUTER_URL", "http://anomaly-router:80")
HTTP_TIMEOUT  = float(os.getenv("MCP_TOOL_HTTP_TIMEOUT", "8.0"))

_TOOL_META = {t["name"]: t for t in FOLLOWUP_TOOLS}


# ============================================================
# FastMCP server — 표준 MCP tool 정의
# ============================================================
mcp = FastMCP(
    name="fraudecom-agent-tools",
    instructions=(
        "Follow-up retrieval tools for a fraud-detection RAG agent. "
        "Each tool standardizes ONE retrieval pattern (same-cluster expansion, "
        "cross-cluster reroute[, feature comparison]) behind a uniform "
        "interface, so the planner LLM can call them without knowing the "
        "underlying Elasticsearch query shape."
    ),
)


# ── Tool 1: expand_within_cluster ───────────────────────────
@mcp.tool(
    name="expand_within_cluster",
    description=_TOOL_META["expand_within_cluster"]["use_when"],
)
async def expand_within_cluster(
    cluster_id: int,
    top_n: int = 5,
    transaction: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    같은 클러스터에서 정상 이웃을 top-1 -> top-k로 확장 retrieve한다.

    Args:
        cluster_id: 현재 매칭된 클러스터 ID (라우팅은 그대로, 이웃 수만 확장).
        top_n: 확장 후 가져올 이웃 개수 (기본 5; 1~20).
        transaction: (옵션) 원본 transaction dict — 일부 Retriever 구현에서
            re-embedding 없이 캐시된 벡터를 재사용하기 위해 전달.

    Returns:
        {"top_n": int, "neighbors": [...], "cluster_persona_summary": str|None}
    """
    top_n = max(1, min(20, int(top_n)))
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(
                f"{RETRIEVER_URL}/retrieve",
                json={"cluster_id": cluster_id, "top_n": top_n,
                      "transaction": transaction or {}, "mode": "within_cluster"},
            )
            resp.raise_for_status()
            data = resp.json()
        return {
            "top_n": top_n,
            "neighbors": data.get("neighbors", []),
            "cluster_persona_summary": data.get("cluster_persona_summary"),
        }
    except Exception as e:
        logger.warning("expand_within_cluster_retriever_failed", error=str(e))
        # graceful degrade — retriever 미배포 환경에서도 agent loop가 진행되도록
        return {"top_n": top_n, "neighbors": [], "cluster_persona_summary": None,
                "_degraded": True, "_error": str(e)}


# ── Tool 2: reroute_cross_cluster ───────────────────────────
@mcp.tool(
    name="reroute_cross_cluster",
    description=_TOOL_META["reroute_cross_cluster"]["use_when"],
)
async def reroute_cross_cluster(
    cluster_id: int,
    top_n: int = 5,
    transaction: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    매칭된 클러스터 자체가 의심될 때, 다른 후보 클러스터들에서 재검색한다.

    Args:
        cluster_id: 현재(의심되는) 클러스터 ID — 재검색 시 후보에서 제외.
        top_n: 새 클러스터에서 가져올 이웃 개수.
        transaction: (옵션) 원본 transaction dict.

    Returns:
        {"cluster_id": int|None, "neighbors": [...], "cluster_persona_summary": str|None}
    """
    top_n = max(1, min(20, int(top_n)))
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(
                f"{RETRIEVER_URL}/retrieve",
                json={"exclude_cluster_id": cluster_id, "top_n": top_n,
                      "transaction": transaction or {}, "mode": "cross_cluster"},
            )
            resp.raise_for_status()
            data = resp.json()
        return {
            "cluster_id": data.get("cluster_id"),
            "neighbors": data.get("neighbors", []),
            "cluster_persona_summary": data.get("cluster_persona_summary"),
        }
    except Exception as e:
        logger.warning("reroute_cross_cluster_retriever_failed", error=str(e))
        return {"cluster_id": None, "neighbors": [], "cluster_persona_summary": None,
                "_degraded": True, "_error": str(e)}


# ── Tool 3: compare_features (Arm 3, ENABLE_COMPARE_FEATURES일 때만 등록) ──
if ENABLE_COMPARE_FEATURES:
    @mcp.tool(
        name="compare_features",
        description=_TOOL_META["compare_features"]["use_when"],
    )
    async def compare_features(
        transaction: Dict[str, Any],
        nearest_normal: Optional[Dict[str, Any]] = None,
        top_n: int = 3,
    ) -> Dict[str, Any]:
        """
        test transaction과 최근접 정상 이웃의 feature 단위 편차를 계산해
        '왜 이상인지'에 대한 구체적 근거(hint)를 생성한다.

        Args:
            transaction: test transaction의 원본 feature dict.
            nearest_normal: (옵션) 최근접 정상 이웃의 원본 feature dict.
                None이면 z-score 편차로 graceful degrade.
            top_n: 상위 몇 개 feature 편차를 반환할지.

        Returns: {"deltas": [...], "hint": str}
        """
        return compute_feature_comparison(
            transaction, nearest_normal, top_n=max(1, min(10, int(top_n)))
        )


# ============================================================
# REST shim — controller.py(LangGraph node)가 호출하는 단순 HTTP 경로
# ============================================================
# 표준 MCP 프로토콜(tools/list, tools/call via JSON-RPC over streamable-http)
# 과는 별개로, controller.py가 이미 httpx.post(f"{TOOLS_MCP_URL}/tools/
# {tool_name}/call", json=payload) 형태로 호출하도록 작성돼 있으므로, 동일
# tool 함수를 그대로 재사용하는 thin REST 래퍼를 같은 ASGI app에 mount한다.
# (= "MCP server이자 동시에 일반 REST tool 서비스" — 마이그레이션 비용 0)

@mcp.custom_route("/tools/{tool_name}/call", methods=["POST"])
async def call_tool_rest(request: Request) -> JSONResponse:
    tool_name = request.path_params["tool_name"]
    if tool_name not in FOLLOWUP_ACTIONS or tool_name == "accept_verdict":
        return JSONResponse({"error": f"unknown or non-callable tool: {tool_name}"},
                             status_code=404)
    payload = await request.json()
    try:
        # FastMCP의 내부 tool 레지스트리를 통해 호출 — stdio/http 어느
        # transport로 등록됐는지와 무관하게 동일 경로로 실행된다.
        # 반환은 SDK 버전에 따라 ContentBlock list(TextContent의 JSON
        # 직렬화된 .text) 또는 dict(structured content) 둘 다 가능하므로
        # 양쪽을 모두 plain dict로 풀어 REST 소비자(controller.py)에게 준다.
        result = await mcp.call_tool(tool_name, payload)
        return JSONResponse(_unwrap_tool_result(result))
    except Exception as e:
        logger.error("rest_shim_tool_call_failed", tool=tool_name, error=str(e))
        return JSONResponse({"error": str(e)}, status_code=500)


def _unwrap_tool_result(result: Any) -> Dict[str, Any]:
    """
    mcp SDK의 call_tool 반환 형태를 plain dict로 정규화한다.
    실측 형태(mcp==1.27.x): tuple(content_blocks: list[ContentBlock],
    structured: dict) — structured는 {"result": <tool 반환값>} 으로 한 번 더
    감싸져 있다. 향후 SDK가 list만 반환하거나 dict만 반환하는 경우도 방어한다.
    """
    if isinstance(result, tuple) and len(result) == 2:
        _content_blocks, structured = result
        if isinstance(structured, dict):
            return structured.get("result", structured)
        result = _content_blocks  # structured가 비정상이면 content_blocks로 폴백

    if isinstance(result, dict):
        return result.get("result", result)

    if isinstance(result, (list, tuple)):
        for block in result:
            if isinstance(block, TextContent):
                try:
                    return _json.loads(block.text)
                except Exception:
                    return {"result": block.text}
        return {"result": [str(b) for b in result]}

    return {"result": result}


@mcp.custom_route("/tools", methods=["GET"])
async def list_tools_rest(request: Request) -> JSONResponse:
    """REST 소비자를 위한 카탈로그 조회 — agent_tools.build_followup_tool_catalog와 동기화."""
    return JSONResponse({"tools": list(FOLLOWUP_ACTIONS)})


@mcp.custom_route("/health", methods=["GET"])
async def health_rest(request: Request) -> JSONResponse:
    return JSONResponse({"status": "healthy", "service": "agent-tools-mcp",
                         "tools": list(FOLLOWUP_ACTIONS)})


# ============================================================
# Entrypoint — transport 선택 (stdio | http)
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="agent_tools MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="http")
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "8090")))
    args = parser.parse_args()

    if args.transport == "stdio":
        # 로컬 client(예: CLI 디버깅, Claude Desktop류)와 프로세스 stdio로 직결.
        mcp.run(transport="stdio")
    else:
        # GKE 배포 기본 모드 — streamable-http(표준 MCP) + REST shim을
        # 동일 ASGI app(/mcp, /tools/*, /health)으로 함께 서빙.
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

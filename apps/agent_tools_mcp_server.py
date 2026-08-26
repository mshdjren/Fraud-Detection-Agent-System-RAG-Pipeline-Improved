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
from pydantic import ValidationError

import json as _json

import httpx
import structlog
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings   # ★ 추가
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
# ★ [V3] 신규 tool 메타 (구버전 agent_tools와의 하위호환 import)
try:
    from agent_tools import EXTRA_TOOLS, EXTRA_TOOL_NAMES, RESEARCH_MODE
except ImportError:                       # legacy agent_tools.py 배포 시
    EXTRA_TOOLS, EXTRA_TOOL_NAMES, RESEARCH_MODE = [], (), "legacy"

# ★ [stage registry] planner action의 단일 출처. tool 이름/feature 상한을
#   여기서 받아 하드코딩을 없앤다(기존 contextual_knn의 `[:2]`가 registry의
#   MAX_CONTEXTUAL_FEATURES와 따로 놀던 문제).
try:
    from agent_tools import (
        PLANNER_STAGE, planner_stage_actions, planner_action_names,
        MAX_CONTEXTUAL_FEATURES, CONTEXTUAL_CANDIDATE_FIELDS,
    )
    _REGISTRY_OK = True
except ImportError:
    PLANNER_STAGE = "stage1"
    MAX_CONTEXTUAL_FEATURES = int(os.getenv("MAX_CONTEXTUAL_FEATURES", "2"))
    CONTEXTUAL_CANDIDATE_FIELDS = ()
    def planner_stage_actions(stage=None): return ("stop", "global_knn", "contextual_knn")
    def planner_action_names(stage=None): return ("stop", "global_knn", "contextual_knn")
    _REGISTRY_OK = False

logger = structlog.get_logger()

# ============================================================
# Configuration — Retriever/Elasticsearch 호출 대상
# ============================================================
RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://anomaly-retriever:80")
ROUTER_URL    = os.getenv("ROUTER_URL", "http://anomaly-router:80")
HTTP_TIMEOUT  = float(os.getenv("MCP_TOOL_HTTP_TIMEOUT", "8.0"))

# ============================================================
# [attribute_knn] sparse 축 검색 — dense ANN을 경유하지 않는다.
#
#   contextual_knn과의 결정적 차이:
#     contextual = dense 전역 pool 수신 → Python soft rerank
#                  → recall이 dense top-N에 갇힘. nd_ctx >= nd_global 항상 성립.
#     attribute  = exact term filter로 부분집합 선택 → 그 안에서 거리 계산
#                  → dense가 버린 후보를 직접 복구. 부분집합/상위집합 관계가 깨진다.
#
#   진단 대응: cluster_mismatch(라우팅 오류)는 원본 feature 축의 문제이므로
#   같은 축에서 검색하는 attribute가 정합. neighbor_insufficient(이웃 부실)는
#   pool을 넓혀야 하므로 contextual이 정합.
# ============================================================
ATTR_TOP_K       = int(os.getenv("ATTR_TOP_K", "5"))
ATTR_MIN_SUPPORT = int(os.getenv("ATTR_MIN_SUPPORT", "3"))   # 이보다 적으면 degraded

# aug 인덱스는 공백 컬럼명이 text로 잡혀 있고 .keyword 서브필드만 term 가능.
# _update_by_query 정규화 완료 후에는 ATTR_FIELD_MODE=normalized로 전환.
ATTR_FIELD_MODE = os.getenv("ATTR_FIELD_MODE", "auto")   # auto | normalized | legacy

_TOOL_META = {t["name"]: t for t in FOLLOWUP_TOOLS}


# ============================================================
# FastMCP server — 표준 MCP tool 정의
# ============================================================
# ★ [FIX] MCP SDK 의 DNS rebinding 보호 대응.
#   transport_security=None + host 기본값(127.0.0.1) 이면 SDK 가 보호를
#   자동 활성화하고 localhost 계열만 허용한다. 클러스터 내부에서
#   http://agent-tools-mcp:80/mcp 로 붙으면 Host: agent-tools-mcp 가 되어
#   "Invalid Host header" → 421 Misdirected Request 로 거부된다.
#   (REST shim /tools/* 는 이 미들웨어를 타지 않아 mcp 모드만 살아 있었다)
_MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
# After — ★ [FIX] wildcard(":*")는 포트가 명시된 Host 헤더에만 매칭됨.
#   K8s Service 기본 포트(80)로 접속 시 클라이언트가 Host 헤더에서 :80을
#   생략하므로, 포트 없는 bare hostname을 반드시 별도로 추가해야 한다.
_ALLOWED_HOSTS = [h.strip() for h in os.getenv(
    "MCP_ALLOWED_HOSTS",
    "agent-tools-mcp,agent-tools-mcp:*,"
    "agent-tools-mcp.elastic.svc.cluster.local,agent-tools-mcp.elastic.svc.cluster.local:*,"
    "localhost,localhost:*,127.0.0.1,127.0.0.1:*,[::1],[::1]:*"
).split(",") if h.strip()]

mcp = FastMCP(
    name="fraudecom-agent-tools",
    host=_MCP_HOST,                       # ★ 0.0.0.0 명시
    port=int(os.getenv("MCP_PORT", "8080")),
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_ALLOWED_HOSTS,
        allowed_origins=["*"],            # 클러스터 내부 호출 — Origin 없음
    ),
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
    cluster_id: int = -1,
    target_cluster_id: Optional[int] = None,
    top_k: int = 5,
    top_n: int = 5,                              # 하위호환 alias
    embedding: Optional[List[float]] = None,
    vec_index: Optional[str] = None,
    transaction: Optional[Dict[str, Any]] = None,
    agent_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    같은 클러스터에서 정상 이웃을 top-k로 확장 retrieve한다.

    ★ [schema 통일] retriever 실제 RetrieveRequest {embedding, cluster_ids,
      vec_index, top_k}로 호출한다 — 기존 {cluster_id, top_n, mode}는
      retriever에 없는 필드라 422 → no-op였다(controller _call_tool_direct와 정합).
      target_cluster_id는 controller가 실어 보내는 대상 cluster(expand는 현 cluster).

    Returns:
        {"top_k": int, "cluster_id": int, "top1_distance": float|None,
         "neighbors": [...], "cluster_persona_summary": str|None}
    """
    _k = max(1, min(20, int(top_k or top_n)))
    _cid = target_cluster_id if target_cluster_id is not None else cluster_id
    _cluster_ids = [int(_cid)] if _cid is not None and int(_cid) >= 0 else []
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(
                f"{RETRIEVER_URL}/retrieve",
                json={"embedding": embedding, "cluster_ids": _cluster_ids,
                      "vec_index": vec_index, "top_k": _k},
            )
            resp.raise_for_status()
            data = resp.json()
        return {
            "top_k": _k,
            "cluster_id": _cid,
            "top1_distance": data.get("top1_distance"),
            "neighbors": data.get("results", data.get("neighbors", [])),
            "cluster_persona_summary": data.get("cluster_persona_summary"),
        }
    except Exception as e:
        logger.warning("expand_within_cluster_retriever_failed", error=str(e))
        return {"top_k": _k, "cluster_id": _cid, "top1_distance": None,
                "neighbors": [], "cluster_persona_summary": None,
                "_degraded": True, "_error": str(e)}


# ── Tool 2: reroute_cross_cluster ───────────────────────────
@mcp.tool(
    name="reroute_cross_cluster",
    description=_TOOL_META["reroute_cross_cluster"]["use_when"],
)
async def reroute_cross_cluster(
    cluster_id: int = -1,
    target_cluster_id: Optional[int] = None,
    top_k: int = 3,
    top_n: int = 5,                              # 하위호환 alias
    embedding: Optional[List[float]] = None,
    vec_index: Optional[str] = None,
    transaction: Optional[Dict[str, Any]] = None,
    agent_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    매칭 클러스터가 의심될 때, controller가 고른 target_cluster_id(사실상 rank2)
    에서 재검색한다.

    ★ [schema 통일 + 실전환] target을 controller가 정하고(router top-5 재사용),
      MCP는 그 cluster로 retriever를 실제 재검색한다 — 기존 {exclude_cluster_id,
      mode}는 retriever에 없어 422 → cluster/nd 불변(no-op)이던 버그의 근본 수정.

    Returns:
        {"top_k": int, "cluster_id": int|None, "top1_distance": float|None,
         "neighbors": [...], "cluster_persona_summary": str|None}
    """
    _k = max(1, min(20, int(top_k or top_n)))
    # target이 없으면 재라우팅 불가 — controller가 반드시 target_cluster_id를 실어야 함.
    _cid = target_cluster_id
    if _cid is None or int(_cid) < 0:
        logger.warning("reroute_no_target", cluster_id=cluster_id,
                       target_cluster_id=target_cluster_id)
        return {"top_k": _k, "cluster_id": None, "top1_distance": None,
                "neighbors": [], "cluster_persona_summary": None,
                "_degraded": True, "_error": "no target_cluster_id"}
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(
                f"{RETRIEVER_URL}/retrieve",
                json={"embedding": embedding, "cluster_ids": [int(_cid)],
                      "vec_index": vec_index, "top_k": _k},
            )
            resp.raise_for_status()
            data = resp.json()
        return {
            "top_k": _k,
            "cluster_id": int(_cid),                 # ★ 실제 전환된 cluster
            "top1_distance": data.get("top1_distance"),
            "neighbors": data.get("results", data.get("neighbors", [])),
            "cluster_persona_summary": data.get("cluster_persona_summary"),
        }
    except Exception as e:
        logger.warning("reroute_cross_cluster_retriever_failed", error=str(e))
        return {"top_k": _k, "cluster_id": None, "top1_distance": None,
                "neighbors": [], "cluster_persona_summary": None,
                "_degraded": True, "_error": str(e)}


# ── Tool V3-1: verify_global — 전역(무필터) HNSW 검증 ────────────
@mcp.tool(
    name="verify_global",
    description="전역(무필터) HNSW top-k 검색 — cluster 제한이 만든 거리 "
                "인플레이션(ε)을 교정해 nd의 진짜 하한과 실제 귀속 cluster를 "
                "확인한다. wm=cm(dominant persona 축 이탈) 시 사용.",
)
async def verify_global(
    top_k: int = 5,
    embedding: Optional[List[float]] = None,
    vec_index: Optional[str] = None,
    transaction: Optional[Dict[str, Any]] = None,
    agent_hint: Optional[str] = None,
    # **_ignored: Any,
    # **kwargs: Any,
) -> Dict[str, Any]:
    """cluster_ids=[] → retriever의 기존 전역 검색 경로 재사용 (retriever 무수정)."""
    _k = max(1, min(20, int(top_k)))
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(
                f"{RETRIEVER_URL}/retrieve",
                json={"embedding": embedding, "cluster_ids": [],
                      "vec_index": vec_index, "top_k": _k},
            )
            resp.raise_for_status()
            data = resp.json()
        res = data.get("results", data.get("neighbors", []))
        return {"top_k": _k, "global": True,
                "cluster_id": (res[0].get("cluster_id") if res else None),  # 실제 귀속지
                "top1_distance": data.get("top1_distance"),
                "neighbors": res,
                "cluster_persona_summary": data.get("cluster_persona_summary")}
    except Exception as e:
        logger.warning("verify_global_retriever_failed", error=str(e))
        return {"top_k": _k, "global": True, "cluster_id": None,
                "top1_distance": None, "neighbors": [],
                "_degraded": True, "_error": str(e)}


# ── Tool GRPO-1: global_knn — 전역(무필터) HNSW (GRPO planner action) ──
# verify_global과 검색 로직은 동일(cluster_ids=[])하지만, 별도 이름/스키마로
# 둔다: verify_global은 legacy/v3 재검색 노드 전용 계약(controller.py의
# _call_tool_direct/_call_tool_mcp*)에 묶여 있어 그 경로를 건드리지 않고
# GRPO planner action(오라클 rollout, controller.py /rollout)을 위한 독립
# 진입점을 만드는 편이 두 planner의 실행 경로를 섞지 않는다.
@mcp.tool(
    name="global_knn",
    description="GRPO planner action. 전역(무필터) HNSW top-k — 현재 cluster/"
                "feature context를 무시하고 전체 정상 memory bank에서 검색한다. "
                "'클러스터를 무시하면 전역적으로 더 적합한 정상 이웃이 있는가?'",
)
async def global_knn(
    top_k: int = 5,
    embedding: Optional[List[float]] = None,
    vec_index: Optional[str] = None,
    # ★ [FIX] registry의 planner_tool_payload가 모든 action에 scope를 실어
    #   보낸다. 파라미터가 없으면 FastMCP 스키마 검증에서 ValidationError →
    #   REST shim 400이 난다. global_knn은 정의상 global 고정이므로 받되 무시.
    scope: str = "global",
    cluster_id: int = -1,
) -> Dict[str, Any]:
    """cluster_ids=[] → retriever 전역 검색 경로. verify_global과 동일 쿼리,
    GRPO action 계약(이름만) 분리."""
    _k = max(1, min(20, int(top_k)))
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(
                f"{RETRIEVER_URL}/retrieve",
                json={"embedding": embedding, "cluster_ids": [],
                      "vec_index": vec_index, "top_k": _k},
            )
            resp.raise_for_status()
            data = resp.json()
        res = data.get("results", data.get("neighbors", []))
        _zero = bool(data.get("zero_hit")) or not res
        return {"top_k": _k, "view": "global", "scope": "global",
                "features_used": [],
                "cluster_id": (res[0].get("cluster_id") if res else None),
                "top1_distance": data.get("top1_distance"),
                "support": data.get("support"),
                "neighbors": res,
                "zero_hit": _zero,
                "_degraded": _zero,
                "_error": ("zero_hit" if _zero else None)}
    except Exception as e:
        logger.warning("global_knn_retriever_failed", error=str(e))
        return {"top_k": _k, "view": "global", "cluster_id": None,
                "top1_distance": None, "neighbors": [],
                "_degraded": True, "_error": str(e)}


# ── Tool GRPO-2: contextual_knn — global scope + soft feature should-boost ──
@mcp.tool(
    name="contextual_knn",
    description="GRPO planner action. Dense embedding similarity는 유지한 "
                "채(hard filter 아님), 지정한 원본 categorical feature가 "
                "일치하는 후보를 soft should-boost로 우선시하는 hybrid "
                "retrieval. features는 evaluator의 agrees/persona_match "
                "후보 중 최대 2개. scope는 V1에서 global로 고정한다(비판적 "
                "검토 §1 — cluster-scoped contextual은 Stage 2 대상).",
)
async def contextual_knn(
    features: Optional[List[str]] = None,
    transaction: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    embedding: Optional[List[float]] = None,
    vec_index: Optional[str] = None,
    # ★ [stage3] scope="cluster"면 cluster 제한을 건 hybrid 검색.
    #   stage1/2 기본값 global에서는 cluster_ids=[]로 기존 동작과 완전히 동일.
    scope: str = "global",
    cluster_id: int = -1,
) -> Dict[str, Any]:
    """
    features 목록을 transaction 값으로 채워 retriever의 should_boost로
    전달한다. transaction에 값이 없는 feature는 조용히 제외한다(빈
    should_boost가 되면 global_knn과 동일해짐 — degraded 플래그로 표시).

    Returns:
        {"top_k", "view":"contextual", "features_used":[...], "cluster_id",
         "top1_distance", "neighbors"}
    """
    _k = max(1, min(20, int(top_k)))
    _txn = transaction or {}
    _should: Dict[str, str] = {}
    # ★ [FIX] 상한을 하드코딩(2)하지 않고 registry의 MAX_CONTEXTUAL_FEATURES를
    #   따른다 — 값이 어긋나면 attribute_knn과 후보 집합이 달라져 비교가 깨진다.
    for f in (features or [])[:MAX_CONTEXTUAL_FEATURES]:
        v = _txn.get(f)
        if v is not None:
            _should[f] = str(v)
    # ★ [stage3] scope 반영. stage1/2에서는 항상 [] (= 전역).
    _cids = [int(cluster_id)] if (scope == "cluster" and int(cluster_id) >= 0) else []
    if not _should:
        logger.warning("contextual_knn_empty_should_boost",
                       features_requested=features, txn_keys=list(_txn)[:12])
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(
                f"{RETRIEVER_URL}/retrieve",
                json={"embedding": embedding, "cluster_ids": _cids,
                      "vec_index": vec_index, "top_k": _k,
                      "should_boost": _should or None,
                      "return_support": True},
            )
            resp.raise_for_status()
            data = resp.json()
        res = data.get("results", data.get("neighbors", []))
        _zero = bool(data.get("zero_hit")) or not res
        return {"top_k": _k, "view": "contextual", "scope": scope,
                "features_used": data.get("features_used") or list(_should.keys()),
                "cluster_id": (res[0].get("cluster_id") if res else None),
                "top1_distance": data.get("top1_distance"),
                # ★ [관측] soft rerank 전 top1 — Δnd 부호 구조 판정용
                "top1_distance_pre_boost": data.get("top1_distance_pre_boost"),
                "support": data.get("support"),
                "neighbors": res,
                "zero_hit": _zero,
                "_degraded": (not bool(_should)) or _zero,
                "_error": ("empty_should_boost" if not _should
                           else ("zero_hit" if _zero else None))}
    except Exception as e:
        logger.warning("contextual_knn_retriever_failed", error=str(e))
        return {"top_k": _k, "view": "contextual", "features_used": [],
                "cluster_id": None, "top1_distance": None, "neighbors": [],
                "_degraded": True, "_error": str(e)}


def _attr_filters(features: Optional[List[str]],
                  transaction: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """★ [FIX — 계약 통일] feature '이름' 리스트 + transaction dict를 받아
    {name: value} term filter를 만든다.

    기존 시그니처는 `features: Dict[name, value]`였는데, 호출자(controller
    _execute_candidate / planner_tool_payload)는 contextual_knn과 동일하게
    `features: List[str]` + `transaction`을 보낸다. 그대로 두면 list에
    .items()를 호출해 AttributeError가 나거나, FastMCP 스키마 검증에서
    422가 난다 — attribute_knn이 오라클에서 통째로 실패했을 경로다.

    두 tool이 같은 payload를 받아야 planner가 soft(contextual) vs
    hard(attribute)를 대칭적으로 다룰 수 있다. 값 해석은 tool 안에서 한다.

    하위호환: features가 dict로 오면(구버전 호출자) 그대로 사용한다.
    """
    if isinstance(features, dict):                       # legacy 호출자
        return {str(k): str(v) for k, v in features.items() if v is not None}
    _txn = transaction or {}
    out: Dict[str, str] = {}
    for f in (features or [])[:MAX_CONTEXTUAL_FEATURES]:
        v = _txn.get(f)
        if v is not None:
            out[str(f)] = str(v)
    return out


@mcp.tool(
    name="attribute_knn",
    description="Planner action. 지정한 원본 categorical feature 값이 정확히 "
                "일치하는 정상 샘플만 hard term filter로 좁힌 뒤, 그 부분집합에서 "
                "embedding 거리 top-k를 반환한다. contextual_knn이 dense 전역 "
                "pool을 재정렬하는 것과 달리 dense 순위 밖의 문서를 직접 복구하므로, "
                "dense 축에서 보이지 않던 증거를 가져온다 — 클러스터 라우팅이 "
                "틀린 경우(wm=cluster_mismatch)의 정합 처방.",
)
async def attribute_knn(
    features: Optional[List[str]] = None,
    transaction: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    embedding: Optional[List[float]] = None,
    vec_index: Optional[str] = None,
    scope: str = "global",
    cluster_id: int = -1,
    request_id: str = "",
) -> Dict[str, Any]:
    """
    Args:
        features:    feature 이름 목록 (최대 MAX_CONTEXTUAL_FEATURES).
                     contextual_knn과 동일 계약 — 값은 transaction에서 조회.
        transaction: 원본 거래 dict (feature 값 조회용)
        top_k:       반환 이웃 수
        embedding:   AnoLLM 임베딩
        vec_index:   router가 결정한 인덱스명

    Returns:
        {"top_k","view":"attribute","scope","features_used","cluster_id",
         "top1_distance","support","neighbors","zero_hit","_degraded","_error"}
    """
    _k = max(1, min(20, int(top_k or ATTR_TOP_K)))
    ff = _attr_filters(features, transaction)
    if not ff:
        logger.warning("attribute_knn_no_features", request_id=request_id,
                       features_requested=features,
                       txn_keys=list((transaction or {}))[:12])
        return {"top_k": _k, "view": "attribute", "scope": scope,
                "features_used": [], "cluster_id": None, "top1_distance": None,
                "support": 0, "neighbors": [], "zero_hit": True,
                "_degraded": True, "_error": "no_features"}
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.post(
                f"{RETRIEVER_URL}/retrieve",
                json={"embedding": embedding,
                      "cluster_ids": ([int(cluster_id)]
                                      if (scope == "cluster" and int(cluster_id) >= 0)
                                      else []),
                      "vec_index": vec_index, "top_k": _k,
                      "field_filters": ff,        # ★ hard term filter
                      "attribute_mode": True,     # ★ SEARCH_MODE 무관하게 필터 강제
                      "return_support": True},    # ★ 진짜 support(필터 통과 문서 수)
            )
            r.raise_for_status()
            data = r.json()
        res = data.get("results", data.get("neighbors", []))
        _sup = data.get("support")
        _sup = int(_sup) if _sup is not None else len(res)
        _zero = bool(data.get("zero_hit")) or not res
        # support가 희박하면 "새 증거"가 아니라 노이즈일 수 있다 — 판정은
        # 하지 않고 degraded 플래그로만 표시해 오라클이 사후 분석하게 한다.
        _low = _sup < ATTR_MIN_SUPPORT
        return {"top_k": _k, "view": "attribute", "scope": scope,
                "features_used": list(ff.keys()),
                "cluster_id": (res[0].get("cluster_id") if res else None),
                "top1_distance": data.get("top1_distance"),
                "support": _sup,
                "neighbors": res,
                "zero_hit": _zero,
                "_degraded": _low or _zero,
                "_error": ("zero_hit" if _zero
                           else (f"low_support({_sup})" if _low else None))}
    except Exception as e:
        logger.warning("attribute_knn_retriever_failed", error=str(e),
                       features_used=list(ff.keys()))
        return {"top_k": _k, "view": "attribute", "scope": scope,
                "features_used": list(ff.keys()), "cluster_id": None,
                "top1_distance": None, "support": 0, "neighbors": [],
                "zero_hit": True, "_degraded": True, "_error": str(e)}


async def attribute_support(
    features: Any, vec_index: str,
    transaction: Optional[Dict[str, Any]] = None,
    cluster_ids: Optional[List[int]] = None, request_id: str = "",
) -> int:
    """term filter만 적용했을 때 남는 문서 수. _count는 매우 싸므로(<10ms)
    planner 프롬프트에 조합별 support를 미리 노출해 희박한 조합을 피하게 한다.
    ★ MCP tool이 아니라 controller가 프롬프트 조립 시 직접 부르는 헬퍼."""
    ff = _attr_filters(features, transaction)
    if not ff:
        return 0
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{RETRIEVER_URL}/count",
            json={"vec_index": vec_index, "field_filters": ff,
                  "cluster_ids": cluster_ids or []},
            timeout=15.0)
        r.raise_for_status()
        return int(r.json().get("count", 0))

# ── Tool V3-2: fetch_contrast — 축 고정 대조 증거 1건 ─────────────────
@mcp.tool(
    name="fetch_contrast",
    description="불일치 categorical 축에서 test와 같은 값을 가진 클러스터 내 "
                "정상 top1을 가져온다 — '같은 프로필의 정상' 인용용 대조 증거 "
                "(판정 불개입, 감사 주석 강화). wm=ni 보조.",
)
async def fetch_contrast(
    cluster_id: int = -1,
    axis: str = "",
    value: str = "",
    embedding: Optional[List[float]] = None,
    vec_index: Optional[str] = None,
    # **_ignored: Any,
    # **kwargs: Any,
) -> Dict[str, Any]:
    """retriever field_filters(V3 신규 파라미터)로 축 고정 top-1 검색."""
    if not axis:
        return {"neighbors": [], "_degraded": True, "_error": "no axis"}
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(
                f"{RETRIEVER_URL}/retrieve",
                json={"embedding": embedding,
                      "cluster_ids": [int(cluster_id)] if int(cluster_id) >= 0 else [],
                      "vec_index": vec_index, "top_k": 1,
                      "field_filters": {axis: value}},
            )
            resp.raise_for_status()
            data = resp.json()
        res = data.get("results", [])
        return {"axis": axis, "value": value,
                "contrast_distance": data.get("top1_distance"),
                "neighbors": res}
    except Exception as e:
        logger.warning("fetch_contrast_retriever_failed", error=str(e))
        return {"axis": axis, "neighbors": [], "_degraded": True, "_error": str(e)}


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
    # ★ [FIX] attribute_knn이 빠져 있어 REST shim(TOOL_EXECUTION_MODE=mcp)
    #   경로에서만 404가 나던 버그. planner action은 registry에서 받아
    #   stage가 늘어나도 여기 목록을 다시 손대지 않게 한다.
    _callable = (set(FOLLOWUP_ACTIONS) | set(EXTRA_TOOL_NAMES)
                | set(planner_action_names()) - {"stop"}
                | {"verify_global", "fetch_contrast",
                   "global_knn", "contextual_knn", "attribute_knn"})
    if tool_name not in _callable or tool_name == "accept_verdict":
        return JSONResponse({"error": f"unknown or non-callable tool: {tool_name}"},
                             status_code=404)
    payload = await request.json()
    try:
        result = await mcp.call_tool(tool_name, payload)
        return JSONResponse(_unwrap_tool_result(result))
    except Exception as e:
        # ★ schema 불일치(payload가 tool signature와 안 맞음)와 tool 본문 실패를
        #   구분한다. FastMCP는 둘 다 ToolError로 감싸지만 __cause__가 다르다:
        #     ValidationError -> 호출자(controller) payload 잘못 = 400
        #     그 외           -> 서버/의존 서비스 문제        = 500
        #   (verify_global의 **kwargs가 required 필드로 잡혀 매 호출 500이 뜨던
        #    사고를 로그만 보고 즉시 판별하기 위한 분기)
        _cause = e.__cause__
        if isinstance(_cause, ValidationError):
            logger.error("rest_shim_schema_mismatch", tool=tool_name,
                         sent_keys=sorted(payload.keys()) if isinstance(payload, dict) else None,
                         errors=[{"field": ".".join(str(x) for x in d.get("loc", ())),
                                  "type": d.get("type")} for d in _cause.errors()])
            return JSONResponse(
                {"error": "argument schema mismatch", "tool": tool_name,
                 "detail": [{"field": ".".join(str(x) for x in d.get("loc", ())),
                             "type": d.get("type"), "msg": d.get("msg")}
                            for d in _cause.errors()]},
                status_code=400,
            )
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
    return JSONResponse({
        "tools": list(FOLLOWUP_ACTIONS),
        "planner_stage": PLANNER_STAGE,
        "planner_actions": [a for a in planner_action_names() if a != "stop"],
        "max_contextual_features": MAX_CONTEXTUAL_FEATURES,
        "registry_ok": _REGISTRY_OK,
    })


@mcp.custom_route("/health", methods=["GET"])
async def health_rest(request: Request) -> JSONResponse:
    return JSONResponse({"status": "healthy", "service": "agent-tools-mcp",
                         "tools": list(FOLLOWUP_ACTIONS),
                         "planner_stage": PLANNER_STAGE,
                         "planner_actions": [a for a in planner_action_names()
                                             if a != "stop"],
                         "registry_ok": _REGISTRY_OK})


# ============================================================
# Entrypoint — transport 선택 (stdio | http)
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="agent_tools MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="http")
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "8080")))
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
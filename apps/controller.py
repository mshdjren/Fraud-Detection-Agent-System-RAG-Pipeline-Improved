"""
Controller Service — Planner + Agent Loop (LangGraph StateGraph)
==================================================================
역할:
  (A) Planner — deep_retrieve → analyzer(3-key schema: cls/sufficiency/
      what_missing) 를 받아 "충분한가? 부족하면 어떤 유형인가?"를 진단해
      follow-up tool 1개를 선택한다.
        action ∈ FOLLOWUP_ACTIONS = {accept_verdict, expand_within_cluster,
                                     reroute_cross_cluster [, compare_features]}
  (B) Agent Loop — (A)의 planner를 LangGraph StateGraph의 `plan` node로
      감싸, retrieve → analyze → plan → (expand|reroute) → analyze ... 의
      순환을 명시적 state로 구현한다. accept_verdict 또는 round/budget
      한도에서 종료한다.

설계 원칙:
  ★ LLM(Gemini Flash Lite)은 이 한 자리(boundary case의 tool 선택)에서만 개입
  ★ planner는 tool 카탈로그 description을 읽고 reasoning으로 선택
  ★ analyzer 출력은 노트북(analyzer_finetune_v9, Section 9 3-key 변환,
    Section 11 평가) 과 100% 동일한 schema:
        {"cls": "NORMAL|ABNORMAL",
         "sufficiency": "confident|ambiguous",
         "what_missing": "<=15 words" | null}
    => sufficiency가 "ambiguous"일 때만 follow-up 후보. "confident"면
       accept_verdict로 바로 수렴 (numeric distance band를 모델이 이미
       내부화했다고 보고, planner가 재해석하지 않음).
    => what_missing은 teacher 단계의 자유 텍스트가 남아있을 수 있으나,
       검증 결과(NB03 v3, S4 1-key 실측 포함) 대부분 distance band의
       재진술이다. 따라서 planner 프롬프트는 what_missing의 키워드를
       "신뢰 가능한 신호"가 아니라 "tool 선택을 위한 약한 힌트"로만
       사용하도록 명시한다.
  ★ Closed action space + closed JSON + 결정론적 fallback
  ★ 구버전 analyzer(6-key: classification/confidence/reasoning/
    key_evidence) 호환을 위해 AnalyzerOutput 필드는 모두 Optional로 유지.

제거된 부가 요소: Layer-1 fast/deep LLM plan, LRU 캐시, MCP/Local/Cascading
                  backend 추상화, CostTracker, rate limiter.
  (fast/deep 분기는 orchestrator의 결정론적 pre-router가 담당 — 본 파일의
   pre_router node가 그 역할을 흡수한다.)
"""

import os
import re
import time
import functools
import json
import operator
import uuid
from typing import Dict, List, Optional, Literal, Tuple, TypedDict, Annotated, Any
import itertools          # ★ rollout feature 조합 생성용 (_feature_combos)


from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import structlog
import asyncio
import httpx

# analyzer 출력 스키마 정규화 — local_analyzer.py의 schema가 바뀌어도
# 이 모듈(analyzer_schema.py)만 고치면 되도록 분리됨.
from analyzer_schema import NormalizedAnalysis, normalize_analyzer_output, map_gap_to_tool

# follow-up tool 카탈로그 — agent_tools가 단일 출처. 미배포 시 fallback.
try:
    from agent_tools import build_followup_tool_catalog, FOLLOWUP_ACTIONS, decide_track
    # ★ [GRPO Planner] 오라클 rollout 전용 action space — 기존 FOLLOWUP_ACTIONS
    #   (accept/expand/reroute)와 완전히 별개 카탈로그. /rollout 엔드포인트만 참조.
    from agent_tools import (
        GRPO_ACTIONS, ACTION_COST, CONTEXTUAL_CANDIDATE_FIELDS,
        MAX_CONTEXTUAL_FEATURES, build_grpo_action_space,
    )
    _GRPO_ACTION_SPACE_OK = True
except Exception:
    FOLLOWUP_ACTIONS = (
        "accept_verdict", "expand_within_cluster", "reroute_cross_cluster",
    )
    def build_followup_tool_catalog() -> str:
        return (
            "## AVAILABLE FOLLOW-UP TOOLS (select EXACTLY ONE)\n"
            "- accept_verdict\n- expand_within_cluster\n- reroute_cross_cluster"
        )
    def decide_track(global_distance: float, *, fast_threshold: float) -> str:
        return "fast" if float(global_distance) < float(fast_threshold) else "deep"
    GRPO_ACTIONS = ("stop", "global_knn", "contextual_knn")
    ACTION_COST = {"stop": 0.0, "contextual_knn": 1.2, "global_knn": 1.5}
    CONTEXTUAL_CANDIDATE_FIELDS = ("sex", "source", "browser", "IP_country",
                                   "month_purchase", "weekday_purchase")
    MAX_CONTEXTUAL_FEATURES = 2
    def build_grpo_action_space(agree_like_axes):
        return [{"action": "stop"}, {"action": "global_knn"}]
    _GRPO_ACTION_SPACE_OK = False

# ★ [Planner Serving] stage registry — 학습된 planner의 action을 현재 stage에서
#   실행 가능한 형태로 정규화하는 단일 관문.
#   ※ 이 블록이 없으면 _node_plan_local(_PLANNER_REGISTRY_OK)뿐 아니라
#     route_after_plan(planner_action_names)까지 NameError로 죽는다.
#     구버전 agent_tools.py가 배포된 환경에서도 서버가 살아있도록 fallback을
#     두되, 그 경우 _node_plan_local이 None을 반환해 legacy 경로로 넘어간다.
try:
    from agent_tools import (
        PLANNER_STAGE, planner_stage_actions, planner_action_names,
        resolve_planner_action, planner_tool_payload,
    )
    _PLANNER_REGISTRY_OK = True
except Exception as _e_reg:
    PLANNER_STAGE = "stage1"
    def planner_stage_actions(stage=None):
        return ("stop", "global_knn", "contextual_knn")
    def planner_action_names(stage=None):
        return ("stop", "global_knn", "contextual_knn")
    def resolve_planner_action(action, features=None, scope=None,
                               candidate_pool=None, stage=None):
        return {"valid": False, "reason": "registry_unavailable",
                "key": None, "spec": None, "features": []}
    def planner_tool_payload(spec, **kw):
        return dict(kw)
    _PLANNER_REGISTRY_OK = False

# 표준 MCP client — TOOL_EXECUTION_MODE=mcp_native 경로 전용. 미설치 환경
# (예: requirements에 langchain-mcp-adapters를 아직 안 올린 컨테이너)에서도
# import 자체가 서버 전체를 죽이지 않도록 try/except로 감싼다 — 이 경로가
# 막혀도 direct/mcp(REST shim)는 정상 동작해야 한다.
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    _MCP_NATIVE_OK = True
except Exception:
    _MCP_NATIVE_OK = False


# ============================================================
# Configuration
# ============================================================
LLM_MODEL            = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
LLM_TEMPERATURE      = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS       = int(os.getenv("LLM_MAX_TOKENS", "512"))
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_REGION  = os.getenv("GOOGLE_CLOUD_REGION", "us-west1")

# 서비스 URL (agent loop가 호출하는 외부 마이크로서비스)
RETRIEVER_URL   = os.getenv("RETRIEVER_URL", "http://anomaly-retriever:80")
ANALYZER_URL    = os.getenv("ANALYZER_URL", "http://anomaly-analyzer:80")
# CPU-only GKE(e2-medium)에서 Qwen 추론은 10~30초 소요 가능
# — 기존 10.0 하드코딩으로 타임아웃 발생 확인됨 (total latency ~10s)
ANALYZER_TIMEOUT = float(os.getenv("ANALYZER_TIMEOUT", "120.0"))

# ── sufficiency deterministic 계산 (analyzer 호출 실패 시 fallback 전용) ──
# 정상 경로에서는 local_analyzer.py(prompts.normalize_dist + _suf_from_nd)가
# 이미 정규화된 sufficiency를 응답에 담아 보내므로 여기서 재계산하지 않는다.
# analyzer가 죽었을 때만 이 상수/함수를 쓴다 — NB02 to_chatml의 _suf_from_nd와
# 동일 규칙이되, top1_distance가 raw L2(~1.2~1.7)로 들어오므로 반드시
# 정규화(sigmoid) 후 0.3/0.55와 비교해야 한다 (예전엔 정규화 없이 비교해서
# 항상 confident로 떨어지는 버그가 있었음).
import math

DIST_P50      = float(os.getenv("DIST_P50", "1.448"))
DIST_IQR      = float(os.getenv("DIST_IQR", "0.387"))
BOUNDARY_LOW  = float(os.getenv("SUFFICIENCY_BOUNDARY_LOW",  "0.3"))
BOUNDARY_HIGH = float(os.getenv("SUFFICIENCY_BOUNDARY_HIGH", "0.55"))
# ★ [confident-skip] det 모드에서 sufficiency=confident(밴드 밖) 케이스는
#   analyzer LLM 호출 산출물이 없다 — cls는 nd threshold로 결정되고 wm=null이
#   정답이므로. 호출 자체를 생략해 "LLM은 필요한 곳(ambiguous의 wm)에서만"
#   이라는 agent 시스템의 효율 논지를 실현한다. forwardRAG(전수 LLM 호출)
#   대비 차별점의 구조적 근거. llm 모드 비교실험 시에는 반드시 false.
SKIP_ANALYZER_ON_CONFIDENT = os.getenv("SKIP_ANALYZER_ON_CONFIDENT", "true").lower() == "true"
CLS_ND_THRESHOLD = float(os.getenv("CLS_ND_THRESHOLD", "0.5"))   # analyzer와 동일 키 공유

# ══════════════════════════════════════════════════════════════════════
# ★ [재설계] 판정 nd 계산 — round0 vs round1(재검색) 이원 척도
#   · round0(재검색 없음): raw top-1 distance → nd (보정 없음).
#     PatchCore 논문에서 top-1 raw가 대부분 데이터셋 최고 AUROC → 보정 불필요.
#   · round1(expand/reroute 재검색 후): PatchCore-based reweighting nd.
#     재검색 결과의 top-1이 "이웃 군집 속에 있나(신뢰) vs 혼자 튀나(고립)"를
#     top-k 거리로 보정. min-fusion은 폐기 — 재검색 여부에 따라 둘 중 하나만.
#   ⚠ 두 척도는 분포가 달라 하나의 CLS_ND_THRESHOLD를 공유하면 미스캘리브레이션
#     위험 → agent_batch_inference가 재검색군/비재검색군 분리 AUROC로 진단.
PATCHCORE_ON_RESEARCH = os.getenv("PATCHCORE_ON_RESEARCH", "true").lower() == "true"
# reweighting 방향(부호) A/B — 두 의도가 부호 반대라 config로 실험 분리:
#   "isolation": 원식 그대로. w=1-softmax(d1) → 재검색 top-1이 이웃 대비 고립될수록
#                nd↑(이상도↑). "가까운데도 주변에 정상 이웃이 없다 = 의심".
#   "density":   밀도 보정. 이웃이 조밀하면 그 매칭을 더 신뢰해 nd를 낮춤.
#                "재검색이 좋은 정상 군집을 찾았으면 정상 쪽으로".
PATCHCORE_REWEIGHT_MODE = os.getenv("PATCHCORE_REWEIGHT_MODE", "isolation").lower()
PATCHCORE_TAU = float(os.getenv("PATCHCORE_TAU", os.getenv("EXPAND_PLATEAU_TAU", "0.142")))
#   ↑ 구 EXPAND_PLATEAU_TAU를 승계 — 이제 gate 임계가 아니라 reweighting/spread
#     관측 스케일 파라미터. (plateau_ratio 계산 자체는 유지 — spread 신호 관측용)

# ★ [evaluator 1회] 재검색 후 round1에서는 analyzer(LLM)를 다시 부르지 않는다.
#   forward(LLM 1회) 대비 latency 이득의 전제 — 재검색 결과는 PatchCore nd(거리)로
#   최종 판정. wm의 판정 기여는 이미 기각됐으므로(net gain −14) 재평가는 순손실.
EVALUATOR_ONCE = os.getenv("EVALUATOR_ONCE", "true").lower() == "true"


def _normalize_dist_fallback(d: float) -> float:
    """local_analyzer.py의 prompts.normalize_dist와 동일 수식.
    controller.py는 prompts.py를 mount하지 않으므로(analyzer 다운 시에만
    쓰는 최후 fallback이라 파일 의존성을 새로 추가하지 않기 위해) 수식만
    인라인 복제한다."""
    if DIST_IQR == 0:
        return 0.5
    return 1 / (1 + math.exp(-(d - DIST_P50) / DIST_IQR))
# ── [검증용] Analyzer Mock 토글 ──────────────────────────────────────
# local analyzer 재튜닝 전, agent loop 자체(expand/reroute → re-analyze
# → finalize 흐름)가 의도대로 동작하는지 검증하는 임시 플래그.
#   round 0 → ambiguous (MOCK_ANALYZER_ACTION에 따라 expand 또는 reroute)
#   round 1+ → confident (accept_verdict → finalize)
# ★ 노트북 재튜닝 완료 후 반드시 false로 되돌릴 것.
FORCE_MOCK_ANALYZER = os.getenv("FORCE_MOCK_ANALYZER", "false").lower() == "true"
MOCK_ANALYZER_ACTION = os.getenv("MOCK_ANALYZER_ACTION", "expand")  # "expand" | "reroute"
TOOLS_MCP_URL   = os.getenv("TOOLS_MCP_URL", "http://agent-tools-mcp:80")
FAST_THRESHOLD  = float(os.getenv("FAST_TRACK_DISTANCE_THRESHOLD", "0.01"))
AGENT_MAX_ROUNDS = int(os.getenv("AGENT_MAX_ROUNDS", "2"))  # expand/reroute는 1회만 허용(루프 가드)
# ★ [tool 의도 정합] 기본 retriever inference top-k=3, expand 시 top-k=5.
#   (기존: evidence_top_n+4=5 고정 — expand 전후 evidence 수 차이가 없던 문제)
BASE_TOP_K   = int(os.getenv("BASE_TOP_K",   "3"))
EXPAND_TOP_K = int(os.getenv("EXPAND_TOP_K", "5"))
# ★ [재설계] tool 선택은 planner(what_missing 기반)가 전담한다 —
#   deterministic gate 오버라이드(_mismatch_gate/_expand_gate)는 제거.
#   근거: analyzer가 ambiguous로 판정하면 wm이 어떤 결함(sparse vs mismatch)인지
#   지목 → planner가 그대로 tool 매핑. gate로 wm을 무시하면 "agent가 스스로
#   판단"하는 아키텍처 정합성이 깨진다. (wm의 misroute 예측력이 낮다는 F7 측정은
#   최종 판정을 거리가 하므로 직접 해를 끼치지 않음 — 재검색 후 nd가 판정.)
# tool 실행 경로 — orchestrator_agentic.py의 동일 이름 env var와 의미를 맞춤
# (오케스트레이터와 controller가 같은 토글로 A/B 비교 가능하도록).
#   "direct"    : expand/reroute를 retriever에 직접 HTTP 호출 (agent_tools_mcp_server 미경유)
#   "mcp"       : agent_tools_mcp_server.py의 REST shim(/tools/{name}/call)을 경유
#   "mcp_native": 표준 MCP client(langchain_mcp_adapters, streamable-http JSON-RPC)로
#                 동일 tool을 호출 — "3-axis transport benchmark"의 세 번째 축.
#                 direct/mcp(REST shim)는 평범한 HTTP POST이지만, 이 경로는 진짜
#                 MCP 프로토콜(tools/list 협상 + JSON-RPC envelope)을 타므로
#                 그 자체의 오버헤드를 별도로 측정할 수 있다.
TOOL_EXECUTION_MODE = os.getenv("TOOL_EXECUTION_MODE", "direct")  # "direct" | "mcp" | "mcp_native"

# ══════════════════════════════════════════════════════════════════════
# ★ [Planner Serving] 학습된 planner(0.5B) 서빙 스위치
#   "gemini"        : 기존 Vertex AI planner (LLM 실패 시 deterministic으로 격하)
#   "deterministic" : LLM 없이 wm_type -> tool 규칙만 (legacy)
#   "local"         : local_planner.py(0.5B SFT/GRPO) 호출 — GRPO action space
#   ※ legacy 3개 action(accept/expand/reroute)과 planner 3개 action(stop/
#     global_knn/contextual_knn)은 완전히 다른 공간이다. "local"일 때만
#     node_research 경로를 타고, 나머지는 기존 expand/reroute 경로 그대로다.
PLANNER_BACKEND = os.getenv("PLANNER_BACKEND", "gemini").lower()
PLANNER_URL     = os.getenv("PLANNER_URL", "http://local-planner:80")
PLANNER_TIMEOUT = float(os.getenv("PLANNER_TIMEOUT", "120.0"))

# ★ [결정 5] planner tool transport — legacy TOOL_EXECUTION_MODE와 독립적으로
#   둔다(legacy는 expand/reroute 계약, 이쪽은 global_knn/contextual_knn 계약).
#   빈 값이면 TOOL_EXECUTION_MODE를 그대로 승계한다.
PLANNER_TOOL_MODE = (os.getenv("PLANNER_TOOL_MODE", "").lower() or TOOL_EXECUTION_MODE)

# ★ [결정 1] 재검색 후 nd 척도 — 오라클(/rollout)은 raw top-1 sigmoid로 reward를
#   계산했다. 서빙에서 PatchCore reweighting을 걸면 학습이 최적화한 척도와
#   판정 척도가 달라진다(= 학습된 정책의 이득이 재현되지 않을 수 있음).
#   "raw_top1"  : 오라클과 동일 척도 (권장, planner 경로 기본값)
#   "patchcore" : legacy 재검색과 동일 척도
#   ※ local-planner-config의 PLANNER_ND_MODE와 같은 값이어야 한다 —
#     다르면 node_research가 경고 로그(planner_nd_mode_mismatch)를 남긴다.
PLANNER_ND_MODE = os.getenv("PLANNER_ND_MODE", "raw_top1").lower()

# ★ [1-step 고정] planner가 담당하는 재검색 단계 수. 현재 학습된 정책은
#   step1만 학습했으므로(노트북 [K3] oracle_df["step"]==1) 서빙도 1로 고정한다.
#   이 값이 2 이상이 되려면 프롬프트(이미 실행한 검색 블록)·오라클 후보 전개·
#   evaluator 재호출 정책이 함께 바뀌어야 한다 — 지금은 하드 가드로 막는다.
PLANNER_MAX_STEPS = int(os.getenv("PLANNER_MAX_STEPS", "1"))

# ★ [pool fallback 정책] candidate feature는 local_analyzer의 evidence
#   (agrees ∪ persona_match)에서만 나온다. 비면 검색 tool이 실행 불가하므로
#   stop으로 종료한다 — categorical 전체로 임의 확장하지 않는다.
#   (임의 확장은 "evaluator가 근거를 못 찾았는데 아무 축으로나 검색"이 되어
#    planner 결정의 근거가 사라지고, 오라클/서빙 pool도 갈라진다.)
#   ★ /rollout(오라클 수집)과 서빙이 같은 값을 봐야 train-serve가 일치한다.
PLANNER_POOL_FALLBACK = os.getenv("PLANNER_POOL_FALLBACK", "false").lower() == "true"
# ★ [V3] 재검색 의미론 — "legacy"(reroute=rank2/expand=k↑) | "v3"(reroute 노드가
#   verify_global(전역 HNSW)로, expand 노드는 PatchCore reweight(기존) + contrast).
#   그래프/enum/노드는 무수정 — 노드 내부 동작만 모드 분기.
RESEARCH_MODE = os.getenv("RESEARCH_MODE", "legacy").lower()
# ★ [V3] typed what_missing 스키마 — analyzer의 wm_type을 planner 매핑에 사용
WM_SCHEMA = os.getenv("WM_SCHEMA", "v3").lower()
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v3").lower()

_WM_TYPED_RE = re.compile(
    r"\s*(cluster_mismatch|neighbor_insufficient|accept)\s*[::]?\s*([^—]*?)\s*(?:—\s*(.*))?$",
    re.IGNORECASE | re.DOTALL)

# axes 토큰에서 필드명만 추출 — "purchase_value z=+0.31" → "purchase_value",
# "IP_country=China" → "IP_country". ★ 원래 대소문자 보존 (transaction 키가
# IP_country / weekday_purchase 처럼 mixed case라 lower()하면 조회가 깨진다).
_AX_HEAD_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)")

def _parse_wm_typed(wm):
    """typed wm → (type, axes, rationale). analyzer가 wm_type을 안 주는 경로 대비.
    axes는 값/z-score 접미를 제거한 순수 필드명 리스트 (NB03 _axis_field와 동일 의도)."""
    if not wm:
        return None, [], None
    m = _WM_TYPED_RE.match(str(wm))
    if not m:
        return None, [], str(wm)
    axes = []
    for tok in (m.group(2) or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        h = _AX_HEAD_RE.match(tok)
        if h:
            axes.append(h.group(1))          # 원본 케이스 유지
    return (m.group(1).lower(),
            axes,
            (m.group(3) or "").strip() or None)


def _resolve_axis(axis, transaction):
    """axes 필드명 → transaction의 실제 키 (대소문자/구분자 무시 매칭).
    모델이 'ip_country'로 써도 transaction의 'IP_country'를 찾아준다."""
    if not axis or not transaction:
        return None
    if axis in transaction:
        return axis
    _n = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
    tgt = _n(axis)
    for k in transaction:
        if _n(k) == tgt:
            return k
    return None

# wm_type → tool (이름은 legacy와 동일 — v3에서는 노드 내부 동작이 다름)
_V3_TOOL_MAP = {"cluster_mismatch": "reroute_cross_cluster",   # v3: 전역 검증(verify_global)
                "neighbor_insufficient": "expand_within_cluster",   # v3: expand→PatchCore reweight
                "accept": "accept_verdict"}

app    = FastAPI(title="Anomaly Planner / Agent Controller", version="4.0.0-langgraph")
logger = structlog.get_logger()

# ============================================================
# Metrics
# ============================================================
plan_requests_total = Counter("plan_requests_total", "Planner requests", ["status"])
plan_action_total   = Counter("plan_action_total",   "Planner action chosen", ["action"])
# 3-axis transport benchmark: direct(REST to retriever) vs mcp(REST shim)
# vs mcp_native(표준 MCP client) 1회 tool 호출의 wall-clock 비용을 동일
# 측정 구간(_call_tool)에서 잰다 — pasted 문서의 "~12.5ms MCP 오버헤드"를
# 추정이 아니라 실측치로 Grafana에서 바로 비교할 수 있게 한다.
mcp_transport_latency = Histogram(
    "mcp_transport_latency_seconds", "Tool transport latency by execution mode",
    ["mode", "tool", "status"],
    buckets=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
)
plan_latency        = Histogram("plan_latency_seconds", "Planner latency",
                                buckets=[0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
detect_requests_total = Counter("detect_requests_total", "Agent /detect requests", ["status", "track"])
detect_latency       = Histogram("detect_latency_seconds", "End-to-end agent loop latency",
                                  buckets=[0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
agent_round_total    = Counter("agent_round_total", "Agent loop rounds executed", ["action"])
agent_path_total     = Counter("agent_path_total", "Full agent path signature per request", ["path"])
agent_loop_budget_exceeded_total = Counter(
    "agent_loop_budget_exceeded_total",
    "Requests forced to finalize by round>=budget without accept_verdict")
agent_rounds_histogram = Histogram(
    "agent_rounds_per_request", "analyze round count per /detect call",
    buckets=[0, 1, 2, 3, 4])


# ============================================================
# I/O Models
# ============================================================

class AnalyzerOutput(NormalizedAnalysis):
    """
    analyzer 응답의 표준 표현. 실제 정규화 로직(어떤 schema인지 판별하고
    어떻게 변환하는지)은 전부 analyzer_schema.py로 이전되었다 — local_
    analyzer.py의 출력 schema가 바뀌면 그 파일의 _detect_xxx/_parse_xxx
    쌍만 추가하면 되고, 이 클래스나 아래의 build_prompt/deterministic_plan
    은 수정할 필요가 없다 (NormalizedAnalysis의 필드만 보기 때문).

    PlanRequest가 FastAPI로 raw JSON body를 받을 때는 _coerce_analyzer
    validator(아래)가 normalize_analyzer_output()을 거쳐 이 클래스를
    구성한다 — 따라서 어떤 schema의 JSON이 와도 검증에서 죽지 않는다.
    """
    class Config:
        extra = "allow"  # analyzer가 알려지지 않은 추가 키를 보내도 죽지 않게

    @property
    def effective_classification(self) -> str:
        """하위호환 별칭 — 기존 코드가 .effective_classification을 참조하던 지점 유지."""
        return self.classification

    @property
    def effective_reasoning(self) -> str:
        """하위호환 별칭 — 기존 코드가 .effective_reasoning을 참조하던 지점 유지."""
        return self.reasoning_text

    # ── 구버전 6키 코드가 .confidence/.reasoning/.key_evidence로 dict 접근하던
    #    패턴을 그대로 쓸 수 있도록 속성 별칭만 유지 (필드 자체는 NormalizedAnalysis에 있음) ──
    @property
    def what_missing(self) -> Optional[str]:
        """하위호환 별칭 — 3키 schema 시절 .what_missing을 참조하던 지점 유지."""
        return self.gap_hint

    @property
    def sufficiency(self) -> str:
        """하위호환 별칭 — 3키 schema 시절 .sufficiency를 참조하던 지점 유지."""
        return "confident" if self.is_sufficient else "ambiguous"

    @property
    def cls(self) -> str:
        """하위호환 별칭 — 3키 schema 시절 .cls를 참조하던 지점 유지."""
        return self.classification


class PlanRequest(BaseModel):
    request_id:              Optional[str] = None
    analyzer:                AnalyzerOutput
    anomaly_score:           float = Field(0.5, ge=0.0, le=1.0)
    top1_distance:           float = 0.0
    distance_band:           str   = "unknown"
    cluster_id:              int   = -1
    cluster_persona_summary: Optional[str] = None
    expansion_used:          bool  = False

    @model_validator(mode="before")
    @classmethod
    def _normalize_analyzer_field(cls, data: Any) -> Any:
        """
        analyzer 필드로 들어온 raw dict가 어떤 schema(3키/6키/그 이후 변형)
        든, AnalyzerOutput으로 직접 매핑을 시도하기 전에 먼저
        analyzer_schema.normalize_analyzer_output()을 거친다.
        이 한 단계 덕분에, local_analyzer.py가 출력 schema를 바꿔도
        PlanRequest/AnalyzerOutput 자체는 손댈 필요가 없다 — 정규화 로직은
        전부 analyzer_schema.py 하나에 있다.
        """
        if isinstance(data, dict) and isinstance(data.get("analyzer"), dict):
            data = dict(data)
            data["analyzer"] = normalize_analyzer_output(data["analyzer"]).model_dump()
        return data


class ExpansionPolicy(BaseModel):
    top_n: int = 5


class PlanResponse(BaseModel):
    request_id:  str
    action:      str = "accept_verdict"           # FOLLOWUP_ACTIONS 중 하나
    verdict:     Optional[Literal["NORMAL", "ABNORMAL"]] = None
    expansion:   ExpansionPolicy = Field(default_factory=ExpansionPolicy)
    agent_hint:  Optional[str] = None
    rationale:   str = ""
    backend:     str = "fallback"
    is_fallback: bool = False
    latency_ms:  float = 0.0


# ── /detect (agent loop entrypoint) I/O ──
class DetectRequest(BaseModel):
    """
    orchestrator_agentic.py 의 /detect 가 호출하는 agent-loop 진입점.
    orchestrator는 pre_router(fast/deep)와 router+retriever "base call"
    한 번만 직접 수행하고, 그 결과(primary_cluster_id/top1_distance/
    cluster_persona_summary 등)를 여기로 그대로 넘긴다 — agent loop의
    반복(analyze→plan→tool→loop)은 전부 이 서비스(LangGraph)가 담당한다.

    base_* 필드가 비어 있으면(=controller를 단독으로 호출하는 경우, 예:
    배치 스모크 테스트) node_retrieve가 자체적으로 RETRIEVER_URL을 호출해
    채운다 — orchestrator 경유 여부와 무관하게 동작하도록 하는 fallback.
    """
    request_id:    Optional[str] = None
    transaction:   Dict = Field(default_factory=dict)
    embedding:     Optional[List[float]] = None
    global_distance: float = 0.5      # pre-router용 global centroid distance(참고용, controller는 재사용만)
    test_idx:      Optional[int] = None

    # ── orchestrator가 선행 수행한 router+retriever base call 결과 (옵셔널) ──
    base_cluster_id:        Optional[int] = None
    base_top1_distance:      Optional[float] = None
    base_distance_band:       Optional[str] = None
    base_cluster_persona_summary: Optional[str] = None
    base_vec_index:            Optional[str] = None
    base_match_type:            Optional[str] = None
    base_top_k_clusters:         Optional[List[Dict]] = None   # [{cluster_id, rank, score, persona}, ...]
    base_persona:                 Optional[Dict] = None          # ★ primary cluster persona dict
    base_match_type:              Optional[str] = None           # ★ "strict*" | "centroid" — reroute 게이트 신호
    base_neighbors:               Optional[List[Dict]] = None  # retriever top_k 결과 → AnalyzeRequest.top_k_normal_results


class DetectResponse(BaseModel):
    request_id:        str
    track:              str               # "fast" | "deep" (controller 입장에서는 항상 "deep")
    classification:     str
    confidence:          Optional[int] = None
    top_1_distance:      Optional[float] = None
    primary_cluster_id:  Optional[int] = None
    match_type:          Optional[str] = None
    reasoning:            str = ""
    planner_used:         bool = False
    agent_trace:          List[str] = Field(default_factory=list)
    latency_ms:           float = 0.0
    tool_execution_mode:   str = "direct"
    # ★ [Observability ③] round별 구조화 로그 + 종료 사유.
    # harness(Layer3)가 verdict change / 경로 기여 / edge 정확성 계산에 사용.
    agent_rounds:          List[Dict] = Field(default_factory=list)
    termination:           Optional[str] = None


# ============================================================
# Vertex AI Client
# ============================================================
try:
    from langchain_google_vertexai import ChatVertexAI
    # langchain 1.x부터 ChatPromptTemplate이 langchain.prompts에서
    # langchain_core.prompts로 이동했다(langchain.prompts는 더 이상 존재
    #하지 않음). 신버전을 우선 시도하고, 구버전 langchain(<1.0) 환경에서도
    # 동작하도록 구경로를 fallback으로 둔다.
    try:
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError:
        from langchain.prompts import ChatPromptTemplate
    _LANGCHAIN_OK = True
except Exception:
    _LANGCHAIN_OK = False
    ChatVertexAI = None
    ChatPromptTemplate = None

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "action":    {"type": "string", "enum": list(FOLLOWUP_ACTIONS)},
        "verdict":   {"type": "string", "enum": ["NORMAL", "ABNORMAL"]},
        "expansion": {
            "type": "object",
            "properties": {"top_n": {"type": "integer", "minimum": 1, "maximum": 20}},
        },
        "agent_hint": {"type": ["string", "null"]},
        "rationale":  {"type": "string"},
    },
    "required": ["action", "rationale"],
}

vertexAI = None
if _LANGCHAIN_OK:
    try:
        vertexAI = ChatVertexAI(
            model_name=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
            streaming=False,
            # convert_system_message_to_human 제거됨 — 이건 PaLM 시절
            # system message 미지원 모델을 위한 호환 파라미터였다. Gemini
            # 계열(gemini-2.5-flash-lite 등)은 system message를 네이티브로
            # 지원하므로 불필요하다.
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_REGION,
            # response_mime_type/response_schema를 model_kwargs가 아니라
            # 1급 파라미터로 직접 전달 — langchain-google-vertexai 3.1.1의
            # ChatVertexAI가 이 두 필드를 pydantic model_fields로 직접
            # 지원하므로(model_kwargs 경유 시 "should be specified
            # explicitly" UserWarning 발생, 동작은 동일하지만 로그가 시끄러움).
            response_mime_type="application/json",
            response_schema=PLAN_SCHEMA,
        )
        logger.info("planner_vertex_ai_initialized", model=LLM_MODEL)
    except Exception as e:
        logger.error("planner_vertex_ai_init_failed", error=str(e))
        vertexAI = None

# ============================================================
# Prompt — sufficiency 우선 의사결정 + what_missing → tool 매핑
# ============================================================
# 변경 핵심 (노트북 3-key schema 반영):
#   1) analyzer가 distance band를 이미 sufficiency로 환산해 내놓으므로,
#      planner는 distance(nd)를 재해석하지 않고 sufficiency만 1차로 본다.
#      sufficiency=confident -> 그 즉시 accept_verdict (무조건 수렴).
#      sufficiency=ambiguous일 때만 what_missing을 읽고 tool을 고른다.
#   2) what_missing이 sparse-evidence 계열 표현이면 expand,
#      cluster/persona mismatch 표현이면 reroute로 매핑한다.
#   3) what_missing이 거의 distance band 재진술("boundary distance; expand
#      evidence" 등 고정 템플릿)일 수 있음을 알고 있으므로, 이를 "확정
#      신호"가 아니라 "약한 힌트"로 취급하고, 신호가 모호하면 기본값으로
#      expand_within_cluster(가장 비용이 싼 follow-up)를 선택하게 한다.
#   4) expansion_used=true면 무조건 accept_verdict (loop 방지 가드, 기존 유지).

SYSTEM_MSG = (
    "You are the Planner for a fraud-detection RAG agent. "
    "An analyzer already examined a transaction and produced a closed verdict: "
    "cls (NORMAL/ABNORMAL), sufficiency (confident/ambiguous), and what_missing "
    "(set only when sufficiency=ambiguous; it may be a typed defect label or a "
    "short free-text hint depending on the analyzer version). "
    "The analyzer's sufficiency already encodes the distance-band judgement — "
    "you must NOT re-derive it from the numbers yourself; just read it. "
    "Your job is a single choice: given that verdict, pick exactly ONE action "
    "from the catalog you are shown. "
    "Read what_missing for WHICH evidence is weak, and match that to the tool "
    "whose described purpose addresses it. Prefer what the analyzer actually "
    "states over your own reading of the raw numbers. "
    "When what_missing is absent or names nothing concrete, do not invent a "
    "reason — take the cheapest follow-up. "
    "Do not invent evidence. Keep rationale/hint short. Output closed JSON only."
)


def build_prompt(req: PlanRequest) -> str:
    a = req.analyzer
    tool_catalog = build_followup_tool_catalog()
    persona_summary = req.cluster_persona_summary or "(no profile)"
    return f"""## ANALYZER OUTPUT
- cls:          {a.effective_classification}
- sufficiency:  {a.sufficiency}
- what_missing: {a.what_missing or "null"}

## CONTEXT SIGNALS (secondary; sufficiency already accounts for these —
## do NOT re-threshold on these yourself)
- anomaly_score:  {req.anomaly_score:.4f}
- top1_distance:  {req.top1_distance:.4f}
- distance_band:  {req.distance_band}
- cluster_id:     {req.cluster_id}
- cluster_persona: {persona_summary}
- expansion_used: {req.expansion_used}

{tool_catalog}

## HOW TO DECIDE
Two invariants first — they override everything below:
1. sufficiency == "confident"  -> accept_verdict, verdict=cls. Stop here.
2. expansion_used == true      -> accept_verdict, verdict=cls (loop guard).

Otherwise the retrieval is ambiguous. Read what_missing and choose the one
tool whose stated purpose addresses the defect it reports.

what_missing arrives in one of two shapes; both carry the same information:
- typed  — it opens with an explicit defect label followed by ":" (the label
  comes from a deterministic check upstream, so let it lead your choice)
- free text — a sentence naming the features or the retrieval state that is
  uncertain; infer the defect from what it actually names

Either way, what matters is WHICH evidence is weak:
- too few neighbors, scattered/spread neighbors, or neighbors disagreeing with
  the transaction  -> a tool that widens or refines retrieval in this cluster
- the transaction does not fit this cluster's persona / likely belongs to a
  different cluster  -> a tool that re-routes across clusters
- it claims no actionable defect  -> accept_verdict, verdict=cls

If what_missing is null, empty, or too generic to point at either case, take
the cheapest retrieval-widening tool. Do not manufacture a mismatch that the
analyzer did not state.

## HINT GENERATION
- agent_hint: one imperative sentence telling the chosen tool what to target,
  reusing the concrete feature names what_missing mentioned (if any).
- If action=accept_verdict, set agent_hint=null.

## OUTPUT (closed JSON, no prose outside)
Fields: action, verdict (only when action=accept_verdict),
expansion {{top_n}}, agent_hint, rationale (1 sentence).
"""


# ============================================================
# Fallback (deterministic) — LLM 불가/에러 시 안전하게 종료
# ============================================================

def fallback_plan(req: PlanRequest, *, reason: str) -> PlanResponse:
    plan_requests_total.labels(status="fallback").inc()
    logger.warning("planner_fallback", reason=reason)
    return PlanResponse(
        request_id=req.request_id or str(uuid.uuid4())[:8],
        action="accept_verdict",
        verdict=req.analyzer.effective_classification,
        expansion=ExpansionPolicy(),
        agent_hint=None,
        rationale=f"fallback ({reason}): accept analyzer verdict",
        backend="fallback",
        is_fallback=True,
    )


def deterministic_plan(req: PlanRequest) -> PlanResponse:
    """
    Vertex AI 미초기화 시에도 agent loop가 끝까지 동작하도록, sufficiency
    -> tool 매핑을 코드로도 보장한다(LLM 결과와 동일 규칙, LLM 호출 없음).

    gap_category -> tool 매핑은 analyzer_schema.map_gap_to_tool() 단일
    함수에 고정되어 있다 — 여기서 별도로 키워드 목록을 재정의하지 않는다.
    (재정의하면, analyzer schema가 바뀌어 analyzer_schema.py의 키워드
    분류가 수정될 때 이 함수가 따라가지 못하는 사고가 날 수 있다.)
    """
    a = req.analyzer
    if req.expansion_used or a.is_sufficient:
        return PlanResponse(
            request_id=req.request_id or str(uuid.uuid4())[:8],
            action="accept_verdict",
            verdict=a.effective_classification,
            rationale="deterministic: sufficiency=confident or expansion already used",
            backend="deterministic",
        )
    # ★ [V3] typed wm이 있으면 그 type이 매핑의 단일 출처 (키워드 규칙보다 우선)
    if WM_SCHEMA == "v3":
        _wt, _wax, _wr = _parse_wm_typed(getattr(a, "what_missing", None))
        if _wt in _V3_TOOL_MAP:
            _act = _V3_TOOL_MAP[_wt]
            logger.info("plan_v3_typed", wm_type=_wt, axes=_wax, action=_act)
            return PlanResponse(
                request_id=req.request_id or str(uuid.uuid4())[:8],
                action=_act,
                verdict=(a.effective_classification if _act == "accept_verdict" else None),
                agent_hint=(f"axes: {', '.join(_wax)}" if _wax else None),
                rationale=f"deterministic-v3: wm_type={_wt} -> {_act}",
                backend="deterministic",
            )
        logger.warning("plan_v3_untyped_fallback",          # ★ 파싱 실패 감시
                       wm_preview=str(getattr(a, "what_missing", ""))[:80])
    action = map_gap_to_tool(a)
    return PlanResponse(
        request_id=req.request_id or str(uuid.uuid4())[:8],
        action=action,
        verdict=None,
        agent_hint=f"resolve: {a.gap_hint}" if a.gap_hint else None,
        rationale=f"deterministic: sufficiency=ambiguous, gap_category={a.gap_category} -> {action}",
        backend="deterministic",
    )


# ============================================================
# Core (Planner — single-shot, used by both /plan and the LangGraph plan node)
# ============================================================

async def _call_llm(req: PlanRequest) -> Tuple[Dict, str]:
    if vertexAI is None:
        raise RuntimeError("vertexAI not initialized")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MSG),
        ("human",  "{human_msg}"),
    ])
    chain = prompt_template | vertexAI
    resp = await asyncio.to_thread(chain.invoke, {"human_msg": build_prompt(req)})
    text = resp.content.strip()
    if text.startswith("```"):
        text = text.lstrip("`").lstrip("json").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text), "gemini_flash_lite"


def _normalize(out: Dict, req: PlanRequest, backend: str, latency_ms: float) -> PlanResponse:
    action = out.get("action", "accept_verdict")
    if action not in FOLLOWUP_ACTIONS:
        action = "accept_verdict"

    verdict = out.get("verdict")
    if action == "accept_verdict" and verdict not in ("NORMAL", "ABNORMAL"):
        verdict = req.analyzer.effective_classification

    exp = out.get("expansion", {}) or {}
    expansion = ExpansionPolicy(top_n=max(1, min(20, int(exp.get("top_n", 5)))))

    hint = out.get("agent_hint")
    if action == "accept_verdict":
        hint = None
    elif not isinstance(hint, str):
        hint = None

    plan_action_total.labels(action=action).inc()
    return PlanResponse(
        request_id=req.request_id or str(uuid.uuid4())[:8],
        action=action,
        verdict=verdict if action == "accept_verdict" else None,
        expansion=expansion,
        agent_hint=hint,
        rationale=out.get("rationale", ""),
        backend=backend,
        is_fallback=False,
        latency_ms=latency_ms,
    )


async def run_plan(req: PlanRequest) -> PlanResponse:
    """
    /plan 엔드포인트와 LangGraph의 `plan` node가 공유하는 단일 진입점.
    LLM 우선 시도 -> 실패 시 deterministic_plan (fallback_plan이 아니라
    sufficiency 규칙 그대로 적용하는 deterministic_plan으로 격하해, agent
    loop가 LLM 장애에도 의도된 tool 선택 로직을 유지하도록 한다).
    """
    start = time.time()
    try:
        out, backend = await _call_llm(req)
        latency_ms = (time.time() - start) * 1000
        result = _normalize(out, req, backend, latency_ms)
        plan_requests_total.labels(status="success").inc()
        plan_latency.observe(latency_ms / 1000)
        return result
    except Exception as e:
        logger.warning("planner_llm_failed_using_deterministic", error=str(e))
        result = deterministic_plan(req)
        result.latency_ms = (time.time() - start) * 1000
        plan_requests_total.labels(status="fallback").inc()
        plan_action_total.labels(action=result.action).inc()
        return result


# ============================================================
# LangGraph StateGraph — retrieve -> analyze -> plan -> tool(expand|reroute) -> loop
# ============================================================
try:
    from langgraph.graph import StateGraph, END
    _LANGGRAPH_OK = True
except Exception as e:
    _LANGGRAPH_OK = False
    StateGraph = None
    END = "__end__"
    logger.warning("langgraph_import_failed", error=str(e))


class AgentState(TypedDict, total=False):
    """
    Graph 전체에서 누적되는 state.
    round/budget/history는 loop 제어 + observability(각 round의 state 기록)용.

    ★ 주의: LangGraph는 StateGraph(AgentState) 컴파일 시점에 TypedDict에
    선언된 키만 state 채널로 인식한다. 여기 선언되지 않은 키를 init_state에
    넣어 ainvoke()에 전달하면 조용히 드롭된다(런타임 에러 없이 사라짐) —
    그래서 orchestrator가 넘기는 _base_*/_global_distance 도 전부 정식
    필드로 선언해야 한다 (단순 dict라 동작할 거라 가정하면 안 됨).
    """
    request_id:        str
    track:              str                 # "fast" | "deep" (pre_router 결과)
    transaction:        Dict
    embedding:           Optional[List[float]]
    test_idx:            Optional[int]

    # orchestrator로부터 전달되는 입력 (pre_router/retrieve 판단용, "_" 접두사 = 내부 전용)
    _global_distance:    float
    _base_provided:        bool
    _base_cluster_id:        Optional[int]
    _base_top1_distance:      Optional[float]
    _base_distance_band:       Optional[str]
    _base_cluster_persona_summary: Optional[str]
    _base_neighbors:               Optional[List[Dict]]  # orchestrator가 retriever에서 받은 top_k 결과
    # ★ [tool 의도 정합] retriever 실 API({embedding,cluster_ids,vec_index,top_k})
    # 호출과 reroute의 top2 클러스터 선택에 필요 — orchestrator base call의
    # router 응답(vec_index, top_k_clusters[rank1=percolate top1, rank2~=centroid
    # 거리 score 순])을 그대로 배선한다. percolate version은 건드리지 않는다.
    vec_index:            Optional[str]
    top_k_clusters:        List[Dict]      # [{cluster_id, rank, score, persona}, ...]
    # ★ [persona 전달 복원] 현재 cluster의 persona dict — analyzer의
    # format_persona_zscore 입력. 기존엔 analyze payload에 persona 키 자체가
    # 없어 서빙 local_analyzer가 항상 "no profile"로 돌았다(NB02 학습 입력과
    # 어긋난 train-serve gap). reroute 시 top2 후보의 persona로 교체된다.
    persona:               Optional[Dict]
    match_type:            Optional[str]   # router 매칭 방식 — centroid=percolate 실패

    # retrieve 결과
    top1_distance:       float
    distance_band:       str
    cluster_id:           int
    cluster_persona_summary: Optional[str]
    evidence_top_n:       int               # 현재까지 retrieve한 evidence 개수 (expand가 늘림)
    neighbors:            List[Dict]         # retriever가 반환한 top_k 결과 → AnalyzeRequest.top_k_normal_results

    # analyze 결과 (analyzer의 3-key schema) — 현재 round의 "최신" 값.
    # 기존 노드들(build_prompt, deterministic_plan 등)이 state["analyzer"]를
    # 그대로 참조하므로 하위호환을 위해 유지한다.
    analyzer:             Dict               # AnalyzerOutput.model_dump() 형태

    # ★ 신규: round별 analyzer 출력을 전부 보존하는 채널. operator.add를
    # reducer로 지정했으므로, 노드는 state["analyzer_history"] = [{...}]
    # (이번 round 항목 1개짜리 리스트)만 반환하면 LangGraph가 자동으로
    # 이전 round들의 리스트와 합쳐준다 — 노드 코드가 누적 로직을 직접 짤
    # 필요가 없다. 이게 있으면 "1라운드에서는 sparse, 2라운드 expand 후
    # confident로 바뀜" 같은 추론 궤적(trajectory)을 finalize에서 그대로
    # 들여다볼 수 있어 디버깅/리플레이가 가능해진다(이전엔 "analyzer"
    # 단일 필드가 매 round 덮어써져 1라운드 값이 사라졌었다).
    analyzer_history:     Annotated[List[Dict], operator.add]

    # plan 결과
    action:                str
    verdict:                Optional[str]
    agent_hint:             Optional[str]
    planner_backend:         str

    _zero_hit:               Optional[bool]   # ★ 검색 0건 (TypedDict 미선언 시 LangGraph가 드롭)

    # ★ [Planner Serving] 학습 planner가 고른 action의 부속 정보.
    #   TypedDict에 선언하지 않으면 LangGraph가 조용히 드롭한다(파일 상단 ★주의).
    planner_features:        List[str]
    planner_scope:           Optional[str]
    planner_cost:            Optional[float]
    planner_tool_name:       Optional[str]
    planner_is_fallback:     Optional[bool]
    planner_invalid_reason:  Optional[str]
    planner_source:          Optional[str]   # "model" | "gate" | "fallback"
    planner_gate_reason:     Optional[str]   # 게이트로 stop한 사유(invalid와 구분)

    # loop 제어
    round:                 int
    budget:                 int
    expansion_used:          bool

    # ★ [V3] verify_global / contrast / wm_typed 관련 trace 컬럼 복구 필드
    _nd_global:         Optional[float]
    _flip:              Optional[bool]
    _global_cluster:    Optional[int]
    _contrast_axis:     Optional[str]
    _contrast_distance: Optional[float]
    _contrast_neighbor: Optional[Dict]
    
    # ★ history도 Annotated reducer로 전환. 기존에는 각 노드가
    # state.setdefault("history", []).append(...) 식으로 직접 list를
    # 변경했는데, 이는 "노드가 state를 변경했다"는 부수효과(mutation)에
    # 의존하는 패턴이라 LangGraph의 병렬 브랜치 실행(동시에 여러 노드가
    # 같은 state를 받는 경우) 에서 변경 사항이 충돌하거나 덮어써질 위험이
    # 있다. operator.add reducer로 바꾸면 각 노드는 "이번에 추가할 항목
    # 리스트"만 반환하고, 병합은 LangGraph 프레임워크가 충돌 없이
    # 책임진다.
    history:                  Annotated[List[str], operator.add]

    # ★ [Observability ③] round별 구조화 로그. history(문자열 trace)와 달리
    # 각 round의 판단 근거(sufficiency/what_missing/action/nd)를 dict로
    # 보존한다 — harness가 "경로별 AUROC 기여"와 "conditional edge 정확성"
    # (plan 결정 vs 실제 분기 일치)을 계산하는 유일한 데이터 소스.
    round_log:                Annotated[List[Dict], operator.add]
    # ★ [nd min-fusion] 라운드별 정규화 nd / raw L2 축적 — node_analyze가
    # 라운드마다 1개씩 append, node_finalize가 min을 취해 최종 판정에 사용.
    nd_history:               Annotated[List[float], operator.add]
    raw_dist_history:         Annotated[List[float], operator.add]
    # ★ [재설계] PatchCore reweighting 관측 채널 (재검색 노드가 채움; 일반 채널)
    _last_reweight_w:         Optional[float]   # reweighting 계수 w
    _last_plateau:            Optional[float]   # (d_k−d_1)/d_1 spread
    _nd_source:               Optional[str]     # "raw_top1" | "patchcore_top5"
    # ★ [OBS] 아래 3개는 반드시 선언되어야 한다. TypedDict 에 없는 키를
    #   노드가 반환하면 LangGraph 가 조용히 버려 진단 필드가 사라진다.
    _patchcore_status:        Optional[str]
    _research_error:          Optional[str]
    _research_ms:             Optional[float]   # round1 재검색 소요(ms)
    # 종료 사유: "accept_verdict" | "budget_exceeded" | "fast_track" |
    # "unknown_action" — loop 안전성(budget 초과 0건) 측정용.
    termination:              str


# ============================================================
# 단계별 디버깅 계측 — 각 LangGraph 노드의 입/출력·소요시간·예외를
# request_id 기준으로 일관되게 로그. "어느 단계에서 문제가 생겼는지"를
# 로그 grep 한 번으로 찾기 위한 용도 (직전 agent-loop smoke 디버깅 계기).
# ============================================================
# 로그에 통째로 찍으면 노이즈만 커지는 큰 필드는 제외하고, 나머지는 그대로
# 찍는다 — 필드가 늘어나도 이 셋만 관리하면 됨.
_NODE_LOG_SKIP_KEYS = {
    "transaction", "embedding", "neighbors", "_base_neighbors",
    "history", "analyzer_history",
}


def _compact_state(d: Dict) -> Dict:
    """로그용으로 state/반환값을 축약. 큰 필드는 크기만, 나머지는 값 그대로."""
    out = {}
    for k, v in d.items():
        if k in _NODE_LOG_SKIP_KEYS:
            out[k] = f"<len={len(v)}>" if hasattr(v, "__len__") else v
        else:
            out[k] = v
    return out


def instrument_node(step_name: str, required_keys: tuple = ()):
    """
    LangGraph 노드 함수를 감싸서 step_start/step_done/step_failed를 찍는다.
    노드 로직은 전혀 안 건드리고 데코레이터만 얹으므로 diff가 최소화된다.
    실패 시 step_failed 로그 남긴 뒤 그대로 재raise — 노드 자체의 에러
    핸들링 동작(예: node_analyze의 fallback)은 그대로 유지된다.

    ★ [Observability ③] required_keys: 노드 진입 시 state에 존재해야 하는
    키. TypedDict 미선언 키의 silent drop(주석의 ★주의 케이스)을 런타임에
    잡기 위한 무결성 체크 — 누락 시 raise가 아닌 경고 로그(운영 안전).
    """
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(state: "AgentState") -> "AgentState":
            req_id = state.get("request_id", "?")
            missing = [k for k in required_keys if k not in state]
            if missing:
                logger.warning("state_integrity_violation", step=step_name,
                               request_id=req_id, missing_keys=missing)
            t0 = time.time()
            logger.info("step_start", step=step_name, request_id=req_id,
                        round=state.get("round", 0))
            try:
                result = await fn(state)
                logger.info(
                    "step_done", step=step_name, request_id=req_id,
                    elapsed_ms=round((time.time() - t0) * 1000, 1),
                    result=_compact_state(result) if isinstance(result, dict) else str(type(result)),
                )
                return result
            except Exception as e:
                logger.error(
                    "step_failed", step=step_name, request_id=req_id,
                    elapsed_ms=round((time.time() - t0) * 1000, 1),
                    error_type=type(e).__name__, error=str(e),
                )
                raise
        return wrapper
    return decorator


@instrument_node("pre_router", required_keys=("request_id",))
async def node_pre_router(state: AgentState) -> AgentState:
    """
    결정론적 pre-router: global distance 하나로 fast/deep 분기. LLM 미개입.

    orchestrator_agentic.py가 이미 자신의 /detect에서 fast/deep을 판정해
    deep일 때만 이 서비스를 호출하는 경로(_base_provided=True)에서는,
    여기서 다시 판정하지 않고 무조건 "deep"으로 확정한다 — 이중 판정으로
    fast/deep 임계값이 두 서비스에서 어긋나는 사고를 막기 위함.
    controller를 단독으로 호출하는 경로(스모크 테스트 등)에서는 기존처럼
    global_distance 기준으로 자체 판정한다.
    """
    if state.get("_base_provided"):
        track = "deep"
    else:
        global_distance = state.get("_global_distance", 0.5)  # type: ignore
        track = decide_track(global_distance, fast_threshold=FAST_THRESHOLD)
    # ★ partial dict 반환으로 전환: 이 노드가 실제로 변경한 필드(track)와
    # reducer 채널에 추가할 항목(history)만 반환한다. 이전에는 state
    # 전체(이미 다른 채널들이 채워둔 값까지 포함)를 그대로 return했는데,
    # operator.add reducer가 있는 채널(history, analyzer_history)은 "이
    # 노드가 반환한 값"을 무조건 이전 채널 값에 또 더하기 때문에, 손대지
    # 않은 채널이라도 state 안에 남아있으면 중복 누적이 발생한다(실측
    # 확인됨 — analyzer_history가 plan 노드를 거칠 때마다 2배씩 불어났다).
    # partial dict 반환이 LangGraph의 정식 권장 패턴이기도 하다.
    return {"track": track, "history": ["pre_router"]}


@instrument_node("retrieve", required_keys=("request_id", "transaction"))
async def node_retrieve(state: AgentState) -> AgentState:
    """
    Retriever 호출 → top1_distance/cluster/persona/evidence 갱신.

    orchestrator가 base call(router+retriever)을 이미 수행해 결과를 넘긴
    경우(_base_provided=True), 그래프 진입 직후 첫 retrieve(round==0)는
    HTTP 재호출 없이 해당 결과를 그대로 state에 반영한다 — 동일한
    retrieve를 두 서비스가 중복 수행하지 않도록 함. expand/reroute 이후
    재방문하는 retrieve(round>=1)는 항상 RETRIEVER_URL을 직접 호출한다
    (orchestrator는 최초 base call만 하고, 그 이후 재검색은 agent loop의
    책임이기 때문).
    """
    if state.get("_base_provided") and state.get("round", 0) == 0:
        return {
            "top1_distance": state.get("_base_top1_distance", state.get("top1_distance", 0.5)),
            "distance_band": state.get("_base_distance_band", "unknown"),
            "cluster_id": state.get("_base_cluster_id", state.get("cluster_id", -1)),
            "cluster_persona_summary": state.get("_base_cluster_persona_summary"),
            "neighbors": state.get("_base_neighbors") or [],  # orchestrator base call 결과 재사용
            "history": ["retrieve:base_reused"],
        }

    # ★ retriever 실 API 스키마 ({embedding, cluster_ids, vec_index, top_k})
    _cid = state.get("cluster_id", -1)
    payload = {
        "embedding":   state.get("embedding"),
        "cluster_ids": [_cid] if _cid >= 0 else [],
        "vec_index":   state.get("vec_index"),
        "top_k":       state.get("evidence_top_n", BASE_TOP_K),
    }
    updates: Dict = {"history": ["retrieve"]}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{RETRIEVER_URL}/retrieve", json=payload)
            resp.raise_for_status()
            data = resp.json()
# ★ [zero-hit] retriever가 0건이면 top1_distance=None + zero_hit=true를
        #   반환한다(과거엔 1.0을 지어내 nd=0.24 "confident normal"로 둔갑했다).
        #   여기서는 이전 state 값을 유지하고 로그만 남긴다 — nd를 새로 만들지 않는다.
        _t1 = data.get("top1_distance")
        if _t1 is None or data.get("zero_hit"):
            logger.error("retrieve_zero_hit", cluster_ids=payload["cluster_ids"],
                         vec_index=payload["vec_index"], top_k=payload["top_k"],
                         note="nd 갱신 없이 이전 state 유지")
            updates["_zero_hit"] = True
        else:
            updates["top1_distance"] = float(_t1)
        updates["neighbors"] = data.get("results", [])
        # cluster_id/persona/band는 retriever 응답에 없음 — state 값 유지
    except Exception as e:
        logger.warning("retrieve_failed", error=str(e))
        # retrieve 실패해도 loop는 계속 — 이전 state 값을 변경하지 않음
        # (updates에 안 넣은 필드는 partial dict 반환 시 자동으로 보존됨)
    return updates


@instrument_node("analyze", required_keys=("request_id", "top1_distance"))
async def node_analyze(state: AgentState) -> AgentState:
    """
    Analyzer(Qwen2.5-0.5B sufficiency evaluator) 호출.
    FORCE_MOCK_ANALYZER=true 시 실제 호출 없이 mock 응답 반환.
    """
    current_round = state.get("round", 0)

    # ── [evaluator 1회] 재검색 후 라운드(round>=1)에서는 analyzer(LLM)를
    #    다시 부르지 않는다. 재검색 노드가 이미 PatchCore nd를 nd_history에
    #    기록했고, 그 nd로 finalize가 판정한다. sufficiency=confident를 반환해
    #    plan이 accept_verdict → finalize로 즉시 수렴(LLM 0회 추가).
    if EVALUATOR_ONCE and current_round >= 1:
        _nd = (state.get("nd_history") or [None])[-1]
        if _nd is None:
            _nd = _normalize_dist_fallback(float(state.get("top1_distance", 0.5)))
        raw = {
            "cls": "ABNORMAL" if _nd >= CLS_ND_THRESHOLD else "NORMAL",
            "sufficiency": "confident",          # → planner accept_verdict
            "what_missing": None,
        }
        logger.info("node_analyze_skipped_evaluator_once",
                    request_id=state.get("request_id", "?"),
                    round=current_round, nd=round(float(_nd), 4),
                    nd_source=state.get("_nd_source", "raw_top1"))
        return {
            "analyzer": raw,
            "analyzer_history": [{"round": current_round,
                                  **normalize_analyzer_output(raw).model_dump()}],
            "history": ["analyze:skipped_evaluator_once"],
        }

    # ── [confident-skip] det 정합: nd가 밴드 밖이면 LLM 없이 종결 ──────
    # sufficiency(밴드)와 cls(threshold)는 어차피 deterministic으로 결정되고
    # confident 케이스의 wm 정답은 null — LLM 호출은 정보량 0에 latency만
    # 지불한다. ambiguous 케이스만 analyzer로 보내 wm(재검색 명세)을 얻는다.
    if SKIP_ANALYZER_ON_CONFIDENT and not FORCE_MOCK_ANALYZER:
        _nd_skip = _normalize_dist_fallback(float(state.get("top1_distance", 0.5)))
        if not (BOUNDARY_LOW <= _nd_skip < BOUNDARY_HIGH):
            raw = {
                "cls": "ABNORMAL" if _nd_skip >= CLS_ND_THRESHOLD else "NORMAL",
                "sufficiency": "confident",
                "what_missing": None,
            }
            logger.info("node_analyze_skipped_confident",
                        request_id=state.get("request_id", "?"),
                        round=current_round, nd=round(_nd_skip, 4))
            return {
                "analyzer": raw,
                "analyzer_history": [{"round": current_round,
                                      **normalize_analyzer_output(raw).model_dump()}],
                "history": [f"analyze:skipped_confident(nd={_nd_skip:.3f})"],
                "nd_history": [round(_nd_skip, 4)],                       # ★ min-fusion 소스
                "raw_dist_history": [float(state.get("top1_distance", DIST_P50))],
            }

    # ── [검증용] Mock Analyzer ──────────────────────────────────────────
    # local analyzer 재튜닝 전 agent loop 자체를 검증하기 위한 임시 분기.
    # round 0: ambiguous → MOCK_ANALYZER_ACTION(expand/reroute) 유도
    # round 1+: confident → accept_verdict → finalize 진행
    if FORCE_MOCK_ANALYZER:
        if current_round == 0:
            # 첫 번째 analyze: 재검색이 필요한 것처럼 ambiguous 강제
            what_missing = (
                "only top-1 seen, need more neighbors"
                if MOCK_ANALYZER_ACTION == "expand"
                else "persona mismatch, wrong cluster suspected"
            )
            raw = {"cls": "ABNORMAL", "sufficiency": "ambiguous", "what_missing": what_missing}
            logger.info("node_analyze_mock",
                        round=current_round, action=MOCK_ANALYZER_ACTION,
                        sufficiency="ambiguous", what_missing=what_missing)
        else:
            # 두 번째 analyze: 재검색 후 충분한 증거를 확보한 것처럼 confident 강제
            raw = {"cls": "ABNORMAL", "sufficiency": "confident", "what_missing": None}
            logger.info("node_analyze_mock",
                        round=current_round, sufficiency="confident")
        return {
            "analyzer": raw,
            "analyzer_history": [{"round": current_round, **normalize_analyzer_output(raw).model_dump()}],
            "history": ["analyze"],
        }

    payload = {
        "test_transaction":     state.get("transaction", {}),
        "cluster_id":           state.get("cluster_id", -1),
        # ★ 거리 신호(plateau)는 base 5개로 계산하되, prompt evidence는
        # evidence_top_n개만 노출 — round0=3(학습 입력 정합), expand 후=5.
        "top_k_normal_results": (state.get("neighbors") or [])[:state.get("evidence_top_n", BASE_TOP_K)],
        "top1_distance":        state.get("top1_distance", 0.5),
        "persona":              state.get("persona"),   # ★ persona z-score 블록 복원
    }
    # ── [진단] analyzer 호출 직전 payload 확인 ──────────────────────
    # top_k_count=0 이면 neighbors가 state에 없는 것 → node_retrieve 문제
    # top_k_count>0 이어도 analyze가 confident를 반환하면 → 모델/프롬프트 문제
    _nb0 = (payload["top_k_normal_results"] or [{}])[0]
    _has_feat = bool(_nb0.get("features")) or any(
        k not in ("es_doc_id", "original_index", "cluster_id", "distance", "score")
        for k in _nb0)
    logger.info(
        "node_analyze_calling",
        cluster_id=payload["cluster_id"],
        top_k_count=len(payload["top_k_normal_results"]),
        neighbor_has_features=_has_feat,        # ★ False면 ni 판정 불가 상태
        top1_distance=payload["top1_distance"],
        round=state.get("round", 0),
    )
    try:
        _t0 = time.time()
        async with httpx.AsyncClient(timeout=ANALYZER_TIMEOUT) as client:
            resp = await client.post(f"{ANALYZER_URL}/analyze", json=payload)
            resp.raise_for_status()
            raw = resp.json()
        # ── [진단] analyzer 응답 raw 확인 ───────────────────────────
        # analyzer가 실제로 반환한 값 — controller의 normalize 전 원본
        # cls/sufficiency/what_missing 키가 있는지, 값이 예상 범위인지
        logger.info(
            "node_analyze_response",
            raw_keys=list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__,
            cls=raw.get("cls") if isinstance(raw, dict) else None,
            sufficiency=raw.get("sufficiency") if isinstance(raw, dict) else None,
            what_missing=raw.get("what_missing") if isinstance(raw, dict) else None,
            round=state.get("round", 0),
            analyzer_call_ms=round((time.time() - _t0) * 1000, 1),
        )
    except httpx.TimeoutException as e:
        # ANALYZER_TIMEOUT(기본 120s) 안에 응답을 못 받은 경우 — GGUF 추론이
        # 너무 오래 걸리거나(CPU throttling), analyzer pod가 재시작 중일 때.
        logger.warning(
            "analyze_failed_timeout", error=str(e),
            elapsed_ms=round((time.time() - _t0) * 1000, 1), timeout_s=ANALYZER_TIMEOUT,
        )
        raw = None
    except httpx.HTTPStatusError as e:
        # analyzer가 4xx/5xx를 반환 — 422면 모델 출력 JSON 파싱 실패,
        # 5xx면 analyzer 내부 예외. 본문(detail)까지 로그로 남긴다.
        logger.warning(
            "analyze_failed_http_error", status_code=e.response.status_code,
            body=e.response.text[:500], elapsed_ms=round((time.time() - _t0) * 1000, 1),
        )
        raw = None
    except Exception as e:
        # connection refused(pod 재시작 중), DNS 실패 등 그 외 모든 경우.
        logger.warning(
            "analyze_failed_other", error_type=type(e).__name__, error=str(e),
            elapsed_ms=round((time.time() - _t0) * 1000, 1),
        )
        raw = None

    if raw is None:
        raw_dist = state.get("top1_distance", DIST_P50)   # raw L2, 정규화 전
        nd  = _normalize_dist_fallback(raw_dist)           # ★ 정규화 후 비교 (버그 수정 지점)
        suf = "ambiguous" if BOUNDARY_LOW <= nd < BOUNDARY_HIGH else "confident"
        raw = {
            "cls": "ABNORMAL" if nd >= CLS_ND_THRESHOLD else "NORMAL",   # nd가 이제 정규화된 값이라 정상 비교
            "sufficiency": suf,
            "what_missing": "boundary distance; expand evidence" if suf == "ambiguous" else None,
        }
        logger.warning("analyze_using_distance_fallback", nd=nd, sufficiency=suf)
    normalized = normalize_analyzer_output(raw)
    # ★ node_plan에서 PlanRequest(analyzer=dict)로 model_validator를 통해
    # 재정규화하므로, 여기서 AnalyzerOutput(**dump) 재구성은 불필요.
    # 오히려 model_dump() → AnalyzerOutput(**dump) 를 거치면 raw 필드가
    # 중첩되며 classification이 list로 꼬이는 버그가 재발한다.
    # state["analyzer"]에는 analyzer 원본 응답(raw)을 그대로 저장한다.
    # node_plan의 PlanRequest(analyzer=dict) model_validator가
    # normalize_analyzer_output을 단 한 번만 거쳐 올바르게 처리한다.
    # ★ [nd min-fusion] 이번 라운드의 nd 축적 — analyzer 응답에 nd가 있으면
    #   그 값을(local_analyzer v3가 동봉), 없으면 동일 수식으로 재계산.
    _raw_d  = float(state.get("top1_distance", DIST_P50))
    _nd_now = raw.get("nd") if isinstance(raw, dict) and isinstance(raw.get("nd"), (int, float)) \
              else _normalize_dist_fallback(_raw_d)
    # ★ [EXP] TTFT/TPOT pass-through — normalize_analyzer_output() 이
    #   NormalizedAnalysis 필드만 남기고 버리므로 raw 에서 직접 꺼낸다.
    #   analyzer_history 의 ttft_ms 개수 = 실제 LLM 호출 횟수가 되어
    #   selective invocation 절감률이 자동으로 계산된다.
    _gen = {k: raw[k] for k in ("ttft_ms", "tpot_ms", "n_output_tokens")
            if isinstance(raw, dict) and raw.get(k) is not None}

    return {
        "analyzer": raw,
        "analyzer_history": [{"round": state.get("round", 0),
                              **normalized.model_dump(), **_gen}],
        "history": ["analyze"],
        "nd_history": [round(float(_nd_now), 4)],
        "raw_dist_history": [_raw_d],
    }


@instrument_node("plan", required_keys=("request_id", "analyzer"))
async def node_plan(state: AgentState) -> AgentState:
    # ★ [Planner Serving] PLANNER_BACKEND="local"이면 학습된 0.5B planner를
    #   호출한다. None이 오면(registry 미배포 등) 아래 legacy 경로로 계속 진행.
    if PLANNER_BACKEND == "local":
        _local = await _node_plan_local(state)
        if _local is not None:
            return _local
    analyzer_out = state.get("analyzer", {})
    # ★ analyzer_out을 dict 그대로 넘겨야 PlanRequest의 model_validator
    # (_normalize_analyzer_field)가 normalize_analyzer_output()을 거쳐
    # AnalyzerOutput으로 안전하게 변환한다.
    # 이전에 AnalyzerOutput(**analyzer_out)으로 직접 생성하면 model_dump()에
    # 포함된 raw/schema_name 등 메타 필드가 pydantic 재구성 과정에서 꼬여
    # classification이 list로 들어오는 케이스가 발생했다.
    plan_req = PlanRequest(
        request_id=state.get("request_id"),
        analyzer=analyzer_out,                      # dict 그대로 — validator가 정규화
        top1_distance=state.get("top1_distance", 0.5),
        distance_band=state.get("distance_band", "unknown"),
        cluster_id=state.get("cluster_id", -1),
        cluster_persona_summary=state.get("cluster_persona_summary"),
        expansion_used=state.get("expansion_used", False),
    )
    plan_resp = await run_plan(plan_req)

    # ★ [재설계] tool 선택은 전적으로 planner(what_missing 기반)가 결정한다.
    #   deterministic gate 오버라이드는 제거 — sufficiency=ambiguous면 wm이
    #   지목한 결함 유형에 따라 planner가 expand/reroute를 고른 결과를 그대로 사용.
    #   loop 가드(expansion_used → accept)는 planner 프롬프트 규칙 4에 이미 있고,
    #   route_after_plan의 budget 가드가 추가 안전망.
    _norm0 = plan_req.analyzer
    action_planner = plan_resp.action           # (관측용 — gate 없으니 == 최종 action)
    gate_plateau = gate_nd = gate_ratio = None  # (하위호환 로그 필드; 항상 None)
    agent_round_total.labels(action=plan_resp.action).inc()
    # ★ partial dict 반환. action/verdict/agent_hint/planner_backend는
    # reducer가 없는 일반 채널이라 안전하게 덮어쓴다. history는 이번에
    # 추가할 항목 1개만 — analyzer_history는 이 노드가 건드리지 않으므로
    # 아예 키 자체를 반환하지 않는다(이전 버그: state 전체를 반환하면
    # analyzer_history가 이미 채워진 값으로 state 안에 남아있어, reducer가
    # 그 전체를 "이 노드의 추가분"으로 오인해 매번 2배로 불어났다).
    # ★ [Observability ③] round별 구조화 로그 — plan 노드는 analyzer 결과와
    # plan 결정을 모두 아는 유일한 지점이라 여기서 1회 기록한다.
    # (nd/sufficiency/what_missing/action → harness의 경로별 기여·edge 정확성
    #  ·plan regret 계산 입력. history 문자열 trace는 그대로 유지.)
    _norm = plan_req.analyzer  # model_validator를 거친 AnalyzerOutput
    round_entry = {
        "round":         state.get("round", 0),
        "track":         state.get("track", "deep"),
        "top1_distance": state.get("top1_distance"),
        "nd":            (state.get("nd_history") or [None])[-1],   # ★ 이번 라운드 nd
        "cluster_id":    state.get("cluster_id"),
        "cls":           getattr(_norm, "classification", None),
        "sufficiency":   getattr(_norm, "sufficiency", None),
        "what_missing":  getattr(_norm, "what_missing", None),
        "action":        plan_resp.action,
        "planner_backend": plan_resp.backend,
        "evidence_top_k": state.get("evidence_top_n", BASE_TOP_K),  # ★ 3→5 전이 추적
        # ★ 게이트 관측 (reroute τ 캘리브레이션·plan regret 계측 입력)
        "action_planner": action_planner,   # gate 제거로 == action (하위호환 유지)
        "gate_nd":        None,              # (deprecated — gate 제거)
        "gate_plateau":   None,
        "gate_ratio":     None,
        "match_type":     state.get("match_type"),
        # ★ reweighting 관측 (재검색 노드가 채움; round0은 None)
        "reweight_w":     state.get("_last_reweight_w"),
        "reweight_mode":  PATCHCORE_REWEIGHT_MODE if state.get("expansion_used") else None,
        "research_ms":    state.get("_research_ms"),   # ★ round1 재검색 소요(ms)
        "nd_source":      state.get("_nd_source", "raw_top1"),
        # ★ [V3] typed wm + verify_global/contrast 트레이스 (v2 경로에선 전부 None)
        "wm_type":        (state.get("analyzer") or {}).get("wm_type"),
        "wm_axes":        (state.get("analyzer") or {}).get("wm_axes"),
        "wm_rationale":   (state.get("analyzer") or {}).get("wm_rationale"),
        "nd_global":      state.get("_nd_global"),
        "flip":           state.get("_flip"),
        "global_cluster": state.get("_global_cluster"),
        "contrast_axis":  state.get("_contrast_axis"),
        "contrast_distance": state.get("_contrast_distance"),
        "research_mode":  RESEARCH_MODE,
        # ★ [EXP] Recall@5 산출용 — 이번 라운드 이웃의 original_index 목록.
        #   orchestrator PipelineResponse 는 이웃을 싣지 않으므로, agent_rounds 를
        #   경유해 harness(agent_batch_inference)까지 전달하는 유일한 통로다.
        #   이게 빠지면 retrieval_recall 이 조용히 0.0 으로 나온다.
        "neighbor_oidx":  [int(_n.get("original_index")) for _n in (state.get("neighbors") or [])
                           if isinstance(_n, dict) and _n.get("original_index") is not None],
    }
    return {
        "action": plan_resp.action,
        "verdict": plan_resp.verdict,
        "agent_hint": plan_resp.agent_hint,
        "planner_backend": plan_resp.backend,
        "history": [f"plan:{plan_resp.action}"],
        "round_log": [round_entry],
    }


def _tool_payload(tool_name: str, state: "AgentState") -> Dict:
    """mcp/mcp_native 경로용 payload — direct 경로와 동일 의미
    (expand: 현 cluster·top_k=5 / reroute: top2 cluster·top_k=3).
    기존 키(cluster_id/transaction/agent_hint)는 하위호환으로 유지."""
    if tool_name == "reroute_cross_cluster":
        if RESEARCH_MODE == "v3":
            # ★ [V3] MCP 경로에서는 verify_global tool로 스위칭 (아래 호출부 참고)
            target, top_k = None, EXPAND_TOP_K
        else:
            target, top_k = _pick_reroute_cluster(state), BASE_TOP_K
    else:
        target, top_k = state.get("cluster_id", -1), EXPAND_TOP_K
    return {
        "cluster_id":        state.get("cluster_id", -1),
        "target_cluster_id": target,
        "transaction":        state.get("transaction", {}),
        "embedding":           state.get("embedding"),
        "vec_index":            state.get("vec_index"),
        "top_k":                 top_k,
        "agent_hint":             state.get("agent_hint"),
    }


def _patchcore_nd(neighbors: List[Dict], raw_top1: float) -> Tuple[float, Optional[float], Optional[float]]:
    """재검색(round1) 결과의 top-k 이웃 거리로 PatchCore reweighting nd를 계산.

    반환: (nd, weight, plateau_ratio)
      nd       : 최종 정규화 거리 (CLS_ND_THRESHOLD와 비교되는 판정 score)
      weight   : reweighting 계수 w (관측/로그용; None이면 이웃부족으로 미적용)
      plateau  : (d_k−d_1)/d_1 spread 지표 (관측용, 항상 계산 시도)

    PATCHCORE_REWEIGHT_MODE:
      "isolation" (원식): w = 1 - softmax_1(d) = 1 - exp(d1/τ)/Σexp(di/τ).
          top-1이 이웃 대비 유독 가까우면(고립) w↑ → score↑. base nd를 (1+w)로 키움.
      "density": w를 신뢰도로 사용. 이웃이 조밀(plateau 작음)하면 nd를 낮춰 정상 쪽으로.

    ⚠ 논문 원식의 분모는 최근접 메모리이웃의 근방이나, 서빙 비용 때문에
      test→top-k 거리로 근사한다(= plateau_ratio의 softmax 형태). 주석 명시.
    """
    ds = sorted(float(n["distance"]) for n in neighbors if n.get("distance") is not None)
    base_nd = _normalize_dist_fallback(raw_top1)
    if len(ds) < 2 or ds[0] <= 0:
        return base_nd, None, None                       # 이웃 부족 → 보정 없이 base
    plateau = (ds[-1] - ds[0]) / ds[0]                    # (d_k−d_1)/d_1 spread
    # ★ PatchCore reweighting. 논문 원식은 최근접 메모리이웃 근방으로 정규화하나,
    #   서빙 비용 때문에 test→top-k 거리로 근사한다. τ 스케일 정합을 위해 top-1
    #   기준 상대거리 (d_i − d_1)로 softmax를 구성한다(절대거리는 τ와 스케일 불일치).
    import math as _m
    tau = max(PATCHCORE_TAU, 1e-6)
    rel = [d - ds[0] for d in ds]                         # d_1은 0 (softmax 최대항)
    exps = [_m.exp(-r / tau) for r in rel]
    p1 = exps[0] / sum(exps)                              # top-1의 softmax 질량 (조밀↑=경쟁↑=작음)
    isolation = p1                                        # ★ p1이 클수록 top-1이 이웃 압도=고립
    #   (rel이 크면 다른 항 exp≈0 → p1≈1 = top-1만 존재 = 고립.
    #    rel이 작으면 이웃이 경쟁 → p1 작음 = 조밀=군집 속.)
    if PATCHCORE_REWEIGHT_MODE == "density":
        # 조밀(isolation 작음)하면 신뢰 → nd를 base보다 낮춤. 고립이면 base 유지.
        nd = base_nd * (0.5 + 0.5 * isolation)
    else:  # "isolation" (원식 방향)
        # 고립(isolation 큼)일수록 이상도↑ → nd를 키움.
        nd = min(1.0, base_nd * (1.0 + (isolation - _median_isolation())))
    return float(nd), round(float(isolation), 4), round(float(plateau), 4)


def _median_isolation() -> float:
    """isolation 보정의 중심값(0.5 근방)을 빼 nd가 base 주변에서 대칭 이동하게 함.
    top-k=5 균등 분포 시 p1≈0.2이므로, 완전 고립(1.0)이 아니면 nd를 과도히 키우지 않도록
    0.5를 기준으로 삼는다(휴리스틱; 오프라인 캘리브레이션 시 교체)."""
    return 0.5


def _plateau_ratio(neighbors: List[Dict]) -> Optional[float]:
    """(d5−d1)/d1 — top-k 이웃 raw 거리의 조밀도 (스케일 무관, NB01 rule과
    동일 정의). base 검색이 5개 거리를 제공해야 계산 가능 — 부족하면 None."""
    ds = sorted(float(n["distance"]) for n in neighbors
                if n.get("distance") is not None)
    if len(ds) < 5 or ds[0] <= 0:
        return None
    return (ds[4] - ds[0]) / ds[0]


def _pick_reroute_candidate(state: "AgentState") -> Optional[Dict]:
    """
    reroute 대상 = router top-5 중 현재 cluster가 아닌 최상위 후보(사실상 rank 2).
    router 알고리즘: rank 1 = percolate 매칭, rank 2~ = centroid 거리 score 순
    (router.py _augment_with_centroid). percolate version은 변경하지 않고,
    이미 base call에서 받아둔 top_k_clusters를 재사용한다 — router 재호출 없음.
    ★ 후보 dict 전체를 반환 — cluster_id뿐 아니라 persona(라우터가 top-5
      후보별로 동봉)까지 함께 써서, 전환된 cluster의 persona를 analyzer에
      전달한다 (이전 구현의 "persona 비움" 임시 처리를 대체).
    """
    current = state.get("cluster_id", -1)
    for c in sorted(state.get("top_k_clusters") or [], key=lambda x: x.get("rank", 99)):
        cid = c.get("cluster_id")
        if cid is not None and cid != current:
            return c
    return None


def _pick_reroute_cluster(state: "AgentState") -> Optional[int]:
    c = _pick_reroute_candidate(state)
    return int(c["cluster_id"]) if c else None


async def _call_tool_direct(tool_name: str, state: AgentState) -> Dict:
    """
    TOOL_EXECUTION_MODE=direct — agent_tools_mcp_server를 거치지 않고
    Retriever에 바로 HTTP 호출한다.

    ★ [tool 의도 정합] retriever.py의 실제 RetrieveRequest 스키마
    ({embedding, cluster_ids, vec_index, top_k})로 호출한다 — 기존 payload
    (transaction/top_n/mode/exclude_cluster_id)는 retriever에 존재하지 않는
    필드라 422로 실패하던 경로였다.
      - expand_within_cluster : 같은 cluster에서 top_k=EXPAND_TOP_K(5) 재검색
                                (기본 inference는 BASE_TOP_K=3)
      - reroute_cross_cluster : router top-5의 rank 2 cluster로 전환해
                                top_k=BASE_TOP_K(3) 검색
    """
    if tool_name == "reroute_cross_cluster":
        if RESEARCH_MODE == "v3":
            # ★ [V3] verify_global — target 추측 없이 전역(무필터) HNSW 1회.
            #   cluster 제한이 만든 거리 인플레이션(ε)을 교정 (offline: ε p90=0.206,
            #   recall@3=59.5%). nd는 단조 감소만 가능 → 안전.
            target_cluster, cluster_ids, top_k = None, [], EXPAND_TOP_K
        else:
            target_cluster = _pick_reroute_cluster(state)
            cluster_ids = [target_cluster] if target_cluster is not None else []
            top_k = BASE_TOP_K
    else:  # expand_within_cluster
        target_cluster = state.get("cluster_id", -1)
        cluster_ids = [target_cluster] if target_cluster >= 0 else []
        top_k = EXPAND_TOP_K

    payload: Dict = {
        "embedding":   state.get("embedding"),
        "cluster_ids": cluster_ids,      # [] = 전체 인덱스 검색 (fallback)
        "vec_index":   state.get("vec_index"),
        "top_k":       top_k,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{RETRIEVER_URL}/retrieve", json=payload)
        resp.raise_for_status()
        data = resp.json()
    _res = data.get("results", [])
    if tool_name == "reroute_cross_cluster" and RESEARCH_MODE == "v3":
        # 전역 top1의 소속 cluster = "실제 귀속지" (trace/주석용)
        target_cluster = (_res[0].get("cluster_id") if _res else None)
    return {
        "top_k": top_k,
        "cluster_id": target_cluster,
        "top1_distance": data.get("top1_distance"),
        "neighbors": _res,
        "global": (tool_name == "reroute_cross_cluster" and RESEARCH_MODE == "v3"),
    }


async def _call_tool_mcp(tool_name: str, state: AgentState) -> Dict:
    """
    TOOL_EXECUTION_MODE=mcp — agent_tools_mcp_server.py의 REST shim
    (/tools/{name}/call)을 경유한다. 동일 tool 구현체가 표준 MCP
    streamable-http 경로로도 노출되어 있으므로, 이 경로를 타면 "tool을
    표준 인터페이스 뒤에 둔" 비교군이 된다 (transport overhead 측정 대상).
    """
    payload = _tool_payload(tool_name, state)
    # ★ [V3] reroute action은 MCP의 verify_global tool로 라우팅 (의미 정합)
    _mcp_tool = ("verify_global" if (tool_name == "reroute_cross_cluster"
                                     and RESEARCH_MODE == "v3") else tool_name)
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{TOOLS_MCP_URL}/tools/{_mcp_tool}/call", json=payload)
        resp.raise_for_status()
        return resp.json()


async def _call_tool_mcp_native(tool_name: str, state: AgentState) -> Dict:
    """
    TOOL_EXECUTION_MODE=mcp_native — 진짜 MCP 프로토콜(streamable-http,
    JSON-RPC tools/list + tools/call)로 agent_tools_mcp_server.py를 호출한다.
    direct/mcp(REST shim)와 달리 평범한 HTTP POST가 아니라 MCP client
    세션이 매번 새로 협상(tools/list)까지 거치므로, "3-axis transport
    benchmark"의 세 번째 축(진짜 MCP 오버헤드)을 잰다.

    direct/mcp가 매 호출마다 새 httpx.AsyncClient를 여는 것과 동일하게,
    여기서도 매 호출마다 새 MultiServerMCPClient를 만든다 — 연결 재사용
    여부 차이가 측정값을 오염시키지 않도록 세 경로의 "1회 호출 비용"
    정의를 통일한다.
    """
    if not _MCP_NATIVE_OK:
        raise RuntimeError("langchain_mcp_adapters not installed — mcp_native mode unavailable")

    payload = _tool_payload(tool_name, state)
    # ★ [V3] reroute action은 MCP의 verify_global tool로 스위칭 (REST shim과 정합).
    #   이 스위치가 없어 mcp_native 경로에서 v3인데도 reroute_cross_cluster가
    #   호출되던 것이 verify_global no-op의 근본 원인이었다.
    _mcp_tool = ("verify_global" if (tool_name == "reroute_cross_cluster"
                                     and RESEARCH_MODE == "v3") else tool_name)
    client = MultiServerMCPClient({
        "agent_tools": {"url": f"{TOOLS_MCP_URL}/mcp", "transport": "streamable_http"},
    })
    tools = await client.get_tools()
    tool = next((t for t in tools if t.name == _mcp_tool), None)
    if tool is None:
        raise RuntimeError(f"tool '{_mcp_tool}' not found via MCP tools/list discovery")

    raw = await tool.ainvoke(payload)
    # langchain 도구 호출 결과는 [{"type": "text", "text": "<json string>"}]
    # 형태로 온다(실측 확인됨) — direct/mcp와 동일한 plain dict로 풀어준다.
    _t = raw[0]["text"]
    try:
        return json.loads(_t)
    except json.JSONDecodeError:
        # ★ FastMCP가 tool 예외를 텍스트로 돌려준 경우 — json.loads가
        #   JSONDecodeError를 내면서 진짜 원인이 가려진다. 원문을 남긴다.
        raise RuntimeError(f"mcp_tool_error({tool_name}): {str(_t)[:300]}")
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


async def _call_tool(tool_name: str, state: AgentState) -> Dict:
    """
    TOOL_EXECUTION_MODE(env)에 따라 direct/mcp/mcp_native 실행 경로를
    선택하는 단일 진입점. 세 경로 모두 동일 wall-clock 측정 구간으로
    감싸 mcp_transport_latency_seconds에 기록한다 — Grafana/Prometheus에서
    direct vs mcp vs mcp_native의 1회 호출 비용을 바로 비교할 수 있다
    (pasted 문서의 "~12.5ms MCP 오버헤드" 주장을 데이터로 검증하는 지점).
    """
    t0 = time.time()
    try:
        if TOOL_EXECUTION_MODE == "mcp_native":
            result = await _call_tool_mcp_native(tool_name, state)
        elif TOOL_EXECUTION_MODE == "mcp":
            result = await _call_tool_mcp(tool_name, state)
        else:
            result = await _call_tool_direct(tool_name, state)
        mcp_transport_latency.labels(mode=TOOL_EXECUTION_MODE, tool=tool_name, status="ok").observe(time.time() - t0)
        return result
    except Exception:
        mcp_transport_latency.labels(mode=TOOL_EXECUTION_MODE, tool=tool_name, status="error").observe(time.time() - t0)
        raise


def _apply_patchcore_after_research(updates: Dict, neighbors: List[Dict],
                                    raw_top1: Optional[float]) -> None:
    """재검색(expand/reroute) 결과에 PatchCore reweighting nd를 계산해 updates에 기록.
    round1의 판정 nd는 이 값을 쓴다(min-fusion 폐기 — 재검색했으면 patchcore nd만)."""
    # ★ [OBS] nd 가 안 움직이는 원인을 세 갈래로 구분해 기록한다.
    #   (Δnd=0 이 나올 때 "patchcore off / 이웃부족 / 실제 무변화" 중
    #    어느 쪽인지 로그만 봐서는 알 수 없어 진단이 막혔던 지점)
    if not PATCHCORE_ON_RESEARCH:
        updates["_patchcore_status"] = "off"
        logger.warning("patchcore_skipped", reason="PATCHCORE_ON_RESEARCH=false")
        return
    if raw_top1 is None:
        updates["_patchcore_status"] = "no_raw_top1"
        logger.warning("patchcore_skipped", reason="raw_top1 is None")
        return
    if not neighbors:
        updates["_patchcore_status"] = "no_neighbors"
        logger.warning("patchcore_skipped", reason="neighbors empty")
        return
    nd, w, plateau = _patchcore_nd(neighbors, float(raw_top1))
    if w is None:                                          # len(ds)<2 또는 ds[0]<=0
        updates["_patchcore_status"] = "insufficient_neighbors"
        logger.warning("patchcore_degraded", n_neighbors=len(neighbors),
                       n_with_distance=sum(1 for n in neighbors
                                           if n.get("distance") is not None))
    else:
        updates["_patchcore_status"] = "ok"
    updates["nd_history"] = [round(nd, 4)]                 # ★ 판정 nd (patchcore)
    updates["raw_dist_history"] = [float(raw_top1)]
    updates["_last_reweight_w"] = w
    updates["_last_plateau"] = plateau
    updates["_nd_source"] = "patchcore_top5"



# ============================================================
# ★ [Planner Serving] 학습된 planner의 action 실행 경로
# ============================================================
# legacy(expand/reroute)와 노드/함수를 공유하지 않는다 — payload 계약도
# nd 척도도 다르기 때문. 그래프에는 node_research 하나만 추가된다.

async def _call_planner_tool_direct(tool_name: str, payload: Dict) -> Dict:
    """retriever /retrieve 직접 호출 (agent-tools-mcp 미경유).
    agent_tools_mcp_server의 global_knn/contextual_knn과 동일한 반환 계약을
    만들어 준다 — 세 transport의 응답 스키마가 같아야 node_research가
    분기 없이 동작한다."""
    _scope = payload.get("scope", "global")
    _cids = [int(payload["cluster_id"])] if (_scope == "cluster"
             and payload.get("cluster_id") is not None
             and int(payload["cluster_id"]) >= 0) else []
    body = {"embedding": payload.get("embedding"), "cluster_ids": _cids,
            "vec_index": payload.get("vec_index"),
            "top_k": int(payload.get("top_k", EXPAND_TOP_K))}
    _should: Dict[str, str] = {}
    if tool_name == "contextual_knn":
        _txn = payload.get("transaction") or {}
        for f in (payload.get("features") or [])[:MAX_CONTEXTUAL_FEATURES]:
            if _txn.get(f) is not None:
                _should[f] = str(_txn[f])
        body["should_boost"] = _should or None
    async with httpx.AsyncClient(timeout=PLANNER_TIMEOUT) as client:
        resp = await client.post(f"{RETRIEVER_URL}/retrieve", json=body)
        resp.raise_for_status()
        data = resp.json()
    res = data.get("results", data.get("neighbors", [])) or []
    return {"top_k": body["top_k"],
            "view": "contextual" if tool_name == "contextual_knn" else "global",
            "features_used": list(_should.keys()),
            "cluster_id": (res[0].get("cluster_id") if res else None),
            "top1_distance": data.get("top1_distance"),
            "neighbors": res,
            "_degraded": (tool_name == "contextual_knn" and not _should)}


async def _call_planner_tool_rest(tool_name: str, payload: Dict) -> Dict:
    """agent_tools_mcp_server의 REST shim(/tools/{name}/call). body는 tool
    시그니처 그대로(= MCP native와 동일 payload)."""
    async with httpx.AsyncClient(timeout=PLANNER_TIMEOUT) as client:
        resp = await client.post(f"{TOOLS_MCP_URL}/tools/{tool_name}/call", json=payload)
        resp.raise_for_status()
        return resp.json()


async def _call_planner_tool(tool_name: str, payload: Dict) -> Dict:
    """[결정 5] PLANNER_TOOL_MODE에 따라 transport 선택. legacy _call_tool과
    동일하게 mcp_transport_latency에 기록해 3-axis 비교에 그대로 합류시킨다."""
    t0 = time.time()
    try:
        if PLANNER_TOOL_MODE == "mcp_native":
            result = await _call_grpo_tool_native(tool_name, payload)
        elif PLANNER_TOOL_MODE == "mcp":
            result = await _call_planner_tool_rest(tool_name, payload)
        else:
            result = await _call_planner_tool_direct(tool_name, payload)
        mcp_transport_latency.labels(mode=PLANNER_TOOL_MODE, tool=tool_name,
                                     status="ok").observe(time.time() - t0)
        return result
    except Exception:
        mcp_transport_latency.labels(mode=PLANNER_TOOL_MODE, tool=tool_name,
                                     status="error").observe(time.time() - t0)
        raise


def _apply_planner_nd(updates: Dict, neighbors: List[Dict],
                      raw_top1: Optional[float]) -> None:
    """[결정 1] PLANNER_ND_MODE에 따라 재검색 후 판정 nd를 기록한다.
    patchcore면 legacy와 완전히 같은 함수를 재사용하고, raw_top1이면
    오라클 reward가 쓴 것과 같은 sigmoid(raw top-1)를 그대로 쓴다."""
    if PLANNER_ND_MODE == "patchcore":
        _apply_patchcore_after_research(updates, neighbors, raw_top1)
        return
    if raw_top1 is None:
        updates["_patchcore_status"] = "no_raw_top1"
        return
    updates["nd_history"] = [round(_normalize_dist_fallback(float(raw_top1)), 4)]
    updates["raw_dist_history"] = [float(raw_top1)]
    updates["_nd_source"] = "raw_top1_planner"
    updates["_patchcore_status"] = "off_by_planner_nd_mode"


@instrument_node("research", required_keys=("request_id", "action"))
async def node_research(state: AgentState) -> AgentState:
    """
    학습된 planner가 고른 global_knn / contextual_knn 1회 실행.
    (stop은 route_after_plan에서 곧장 finalize로 가므로 이 노드에 오지 않는다.)
    legacy node_expand/node_reroute는 손대지 않는다 — 두 planner의 실행 경로를
    섞지 않기 위함.
    """
    action = state.get("action", "stop")
    resolved = resolve_planner_action(
        action, features=state.get("planner_features") or [],
        scope=state.get("planner_scope"))
    updates: Dict = {
        "expansion_used": True,
        "round": state.get("round", 0) + 1,
        "history": [f"{action}:{PLANNER_TOOL_MODE}"],
    }
    if not resolved["valid"]:
        # 여기까지 왔는데 무효면 registry/stage 불일치 — 재검색 없이 종료.
        logger.warning("planner_action_unresolvable_at_research",
                       action=action, reason=resolved["reason"], stage=PLANNER_STAGE)
        updates["_research_error"] = f"unresolvable:{resolved['reason']}"
        updates["action"] = "stop"
        return updates

    spec = resolved["spec"]
    _t0 = time.time()
    try:
        payload = planner_tool_payload(
            spec, features=resolved["features"],
            embedding=state.get("embedding"), vec_index=state.get("vec_index"),
            transaction=state.get("transaction") or {},
            cluster_id=state.get("cluster_id", -1), top_k=EXPAND_TOP_K)
        result = await _call_planner_tool(spec["tool_name"], payload)
        _nb = result.get("neighbors") or []
        _raw = result.get("top1_distance")
        if _nb:
            updates["neighbors"] = _nb
        if _raw is not None:
            updates["top1_distance"] = float(_raw)
        updates["evidence_top_n"] = result.get("top_k", EXPAND_TOP_K)
        if spec["scope"] == "global" and result.get("cluster_id") is not None:
            # 전역 검색이 실제로 귀속시킨 cluster — 판정에는 안 쓰고 trace만.
            updates["_global_cluster"] = result.get("cluster_id")
        if result.get("_degraded"):
            updates["_research_error"] = f"degraded:{result.get('_error', 'empty_should_boost')}"
        _apply_planner_nd(updates, _nb or state.get("neighbors") or [],
                          _raw if _raw is not None else state.get("top1_distance"))
        _result_meta = {
            "support":       result.get("support"),
            "view":          result.get("view"),
            "features_used": result.get("features_used"),
            "zero_hit":      bool(result.get("zero_hit")),
            "top1_distance_pre_boost": result.get("top1_distance_pre_boost"),
            "research_cluster_id": result.get("cluster_id"),
        }
    except Exception as e:
        logger.warning("planner_research_failed", action=action,
                       mode=PLANNER_TOOL_MODE, error=str(e))
        updates["_research_error"] = f"{action}:{type(e).__name__}:{str(e)[:120]}"
        updates["_patchcore_status"] = "tool_failed"
        _result_meta = {"zero_hit": True}
    updates["_research_ms"] = round((time.time() - _t0) * 1000, 1)

    # ★ [P1 수정] round_log 엔트리를 남긴다.
    #   기존에는 node_research가 state만 갱신하고 round_log를 안 만들어,
    #   agent_rounds에 round 0과 final만 실렸다. 그 결과 배치 평가에서
    #   재검색 라운드의 neighbor_oidx / support / research_ms를 전혀 찾지
    #   못해 evidence_novelty가 통째로 null이 됐다("재검색이 무엇을 가져왔는가"
    #   를 관측할 수 없는 상태). node_expand/node_reroute는 이미 남기고 있다.
    _nb_final = updates.get("neighbors") or state.get("neighbors") or []
    _round_no = updates["round"]
    updates["round_log"] = [{
        "round":          _round_no,
        "track":          state.get("track", "deep"),
        "action":         action,
        "planner_features": state.get("planner_features") or [],
        "planner_scope":  state.get("planner_scope"),
        "planner_cost":   state.get("planner_cost"),
        "tool_mode":      PLANNER_TOOL_MODE,
        "top1_distance":  updates.get("top1_distance", state.get("top1_distance")),
        "nd":             (updates.get("nd_history") or [None])[-1],
        "nd_before":      (state.get("nd_history") or [None])[-1],
        "nd_source":      updates.get("_nd_source"),
        "cluster_id":     state.get("cluster_id"),
        "evidence_top_k": updates.get("evidence_top_n", EXPAND_TOP_K),
        "research_ms":    updates["_research_ms"],
        "research_error": updates.get("_research_error"),
        "global_cluster": updates.get("_global_cluster"),
        "research_mode":  RESEARCH_MODE,
        # ★ evidence novelty 입력 — 배치 평가가 round0의 것과 대조한다
        "neighbor_oidx":  [int(_n.get("original_index")) for _n in _nb_final
                           if isinstance(_n, dict)
                           and _n.get("original_index") is not None],
        **_result_meta,
    }]
    return updates


async def _node_plan_local(state: AgentState) -> Optional[Dict]:
    """
    PLANNER_BACKEND="local"일 때 node_plan을 대체한다.
    반환 None이면 호출자가 legacy 경로(run_plan)로 계속 진행한다 —
    registry 미배포 같은 이상 상황에서 파이프라인을 멈추지 않기 위함.
    """
    if not _PLANNER_REGISTRY_OK:
        logger.error("planner_registry_unavailable_falling_back_to_legacy")
        return None

    # ★ [1-step 고정] 재검색을 이미 했으면 planner를 부르지 않는다.
    #   (local_planner도 같은 게이트를 갖지만, HTTP 왕복 자체를 없앤다)
    if state.get("expansion_used") or state.get("round", 0) >= PLANNER_MAX_STEPS:
        return {
            "action": "stop",
            # ★ [P5 수정] 게이트 사유를 planner_backend에 섞지 않는다.
            #   섞으면 배치 평가의 backends_seen이 여러 개로 보여
            #   "run 중 ConfigMap이 바뀌었다"는 오탐 경고가 뜬다.
            #   backend는 서빙 백엔드(hf/gguf)만, 사유는 source/gate_reason으로.
            "planner_backend": "local",
            "planner_source": "gate",
            "planner_is_fallback": False,
            "planner_gate_reason": "max_steps_reached",
            "history": ["plan:local:stop(max_steps)"],
        }

    analyzer_raw = state.get("analyzer") or {}
    _norm = normalize_analyzer_output(analyzer_raw)
    _sufficiency = "confident" if _norm.is_sufficient else "ambiguous"

    # ★ [FIX 2] candidate feature의 단일 출처 = _evidence_from_analyzer.
    #   /rollout(오라클)도 같은 함수를 쓰므로 pool 계산 주체가 하나로 모인다.
    #   기존에는 여기서 analyzer_raw를 직접 읽어 두 경로가 갈라질 수 있었다.
    _ev = _evidence_from_analyzer(state)
    _pool = list(dict.fromkeys(_ev["agrees"] + _ev["persona_match"]))
    if not _pool and PLANNER_POOL_FALLBACK:
        _tx = state.get("transaction") or {}
        _pool = [c for c in _CATEGORICAL_FALLBACK if _tx.get(c) is not None]
        logger.info("planner_pool_fallback_applied", n=len(_pool))
    if not _pool:
        # evidence가 비면 검색할 축이 없다 → stop (LLM 호출 없이 종료)
        logger.info("planner_empty_pool_stop", request_id=state.get("request_id"),
                    wm_type=analyzer_raw.get("wm_type"), sufficiency=_sufficiency)
        plan_action_total.labels(action="stop").inc()
        agent_round_total.labels(action="stop").inc()
        return {
            "action": "stop",
            "planner_backend": "local",
            "planner_source": "gate",          # ★ [P5] invalid가 아니라 정상 게이트
            "planner_features": [],
            "planner_is_fallback": False,
            "planner_gate_reason": "empty_candidate_pool",
            "history": ["plan:local:stop(empty_pool)"],
            "round_log": [{
                "round": state.get("round", 0),
                "track": state.get("track", "deep"),
                "top1_distance": state.get("top1_distance"),
                "nd": (state.get("nd_history") or [None])[-1],
                "cluster_id": state.get("cluster_id"),
                "sufficiency": _sufficiency,
                "action": "stop",
                "planner_backend": "local",
                "planner_stage": PLANNER_STAGE,
                "planner_source": "gate",
                "planner_gate_reason": "empty_candidate_pool",
                "planner_nd_mode": PLANNER_ND_MODE,
                "wm_type": analyzer_raw.get("wm_type"),
                "neighbor_oidx": [int(_n.get("original_index"))
                                  for _n in (state.get("neighbors") or [])
                                  if isinstance(_n, dict)
                                  and _n.get("original_index") is not None],
            }],
        }

    payload = {
        "request_id":   state.get("request_id"),
        "test_idx":     state.get("test_idx"),
        "transaction":  state.get("transaction") or {},
        "cluster_id":   state.get("cluster_id", -1),
        "persona":      state.get("persona"),
        "neighbors":    state.get("neighbors") or [],
        "agrees":        _ev["agrees"],
        "persona_match": _ev["persona_match"],
        # ★ 계산된 pool을 명시 전달 — local_planner가 다시 유도하지 않게 한다
        "candidate_features": _pool,
        "nd_before":     (state.get("nd_history") or [None])[-1],
        "top1_distance": state.get("top1_distance"),
        "wm_type":       analyzer_raw.get("wm_type"),
        "sufficiency":   _sufficiency,
        "round":         state.get("round", 0),
        "expansion_used": bool(state.get("expansion_used", False)),
    }
    t0 = time.time()
    resp: Dict = {}
    try:
        async with httpx.AsyncClient(timeout=PLANNER_TIMEOUT) as client:
            r = await client.post(f"{PLANNER_URL}/plan-action", json=payload)
            r.raise_for_status()
            resp = r.json()
    except Exception as e:
        logger.warning("local_planner_call_failed", error_type=type(e).__name__, error=str(e))
        resp = {"action": "stop", "features": [], "is_fallback": True,
                "invalid_reason": f"call_failed:{type(e).__name__}", "source": "fallback"}

    # ★ nd 척도 config가 양쪽에서 어긋나면 학습/판정 척도가 달라진다 — 반드시 경고.
    if resp.get("nd_mode") and resp["nd_mode"] != PLANNER_ND_MODE:
        logger.warning("planner_nd_mode_mismatch",
                       controller=PLANNER_ND_MODE, planner=resp["nd_mode"])
    if resp.get("stage") and resp["stage"] != PLANNER_STAGE:
        logger.warning("planner_stage_mismatch",
                       controller=PLANNER_STAGE, planner=resp["stage"])

    action = resp.get("action", "stop")
    if action not in planner_action_names():
        logger.warning("planner_action_out_of_stage", action=action, stage=PLANNER_STAGE)
        action, resp = "stop", {**resp, "is_fallback": True,
                                "invalid_reason": f"out_of_stage:{action}"}

    latency_ms = (time.time() - t0) * 1000
    plan_action_total.labels(action=action).inc()
    agent_round_total.labels(action=action).inc()
    plan_latency.observe(latency_ms / 1000)
    plan_requests_total.labels(status="fallback" if resp.get("is_fallback") else "success").inc()

    round_entry = {
        "round":         state.get("round", 0),
        "track":         state.get("track", "deep"),
        "top1_distance": state.get("top1_distance"),
        "nd":            (state.get("nd_history") or [None])[-1],
        "cluster_id":    state.get("cluster_id"),
        "cls":           getattr(_norm, "classification", None),
        "sufficiency":   _sufficiency,
        "what_missing":  getattr(_norm, "what_missing", None),
        "action":        action,
        "planner_backend": str(resp.get("backend") or "local"),
        "evidence_top_k": state.get("evidence_top_n", BASE_TOP_K),
        "match_type":     state.get("match_type"),
        "wm_type":        analyzer_raw.get("wm_type"),
        "wm_axes":        analyzer_raw.get("wm_axes"),
        "research_mode":  RESEARCH_MODE,
        "nd_source":      state.get("_nd_source", "raw_top1"),
        # ★ planner 전용 관측 컬럼 (harness/evaluation_metrics가 그대로 집계)
        "planner_stage":       resp.get("stage", PLANNER_STAGE),
        "planner_prompt_ver":  resp.get("prompt_version"),
        "planner_features":    resp.get("features") or [],
        "planner_scope":       resp.get("scope"),
        "planner_cost":        resp.get("cost"),
        "planner_source":      resp.get("source"),
        "planner_is_fallback": bool(resp.get("is_fallback")),
        "planner_invalid_reason": resp.get("invalid_reason"),
        "planner_latency_ms":  round(latency_ms, 1),
        "planner_ttft_ms":     resp.get("ttft_ms"),
        "planner_n_out_tokens": resp.get("n_output_tokens"),
        "planner_nd_mode":     PLANNER_ND_MODE,
        "neighbor_oidx":  [int(_n.get("original_index")) for _n in (state.get("neighbors") or [])
                           if isinstance(_n, dict) and _n.get("original_index") is not None],
    }
    return {
        "action": action,
        "verdict": None,               # 판정은 finalize가 nd로 한다
        "agent_hint": None,
        "planner_backend": str(resp.get("backend") or "local"),
        "planner_features": resp.get("features") or [],
        "planner_scope": resp.get("scope"),
        "planner_cost": resp.get("cost"),
        "planner_tool_name": resp.get("tool_name"),
        "planner_is_fallback": bool(resp.get("is_fallback")),
        "planner_invalid_reason": resp.get("invalid_reason"),
        "history": [f"plan:local:{action}"],
        "round_log": [round_entry],
    }

@instrument_node("expand", required_keys=("cluster_id", "transaction"))
async def node_expand(state: AgentState) -> AgentState:
    _t0 = time.time()
    updates: Dict = {
        "expansion_used": True,
        "round": state.get("round", 0) + 1,
        "history": [f"expand_within_cluster:{TOOL_EXECUTION_MODE}"],
    }
    try:
        # ★ [측정 정확도] expand는 이미 base가 top-5를 갖고 있어 재검색이
        #   불필요하지만, reroute와 동일 조건에서 latency를 재려고 동일 cluster에서
        #   retriever를 실제 재호출한다(top_k=EXPAND_TOP_K).
        result = await _call_tool("expand_within_cluster", state)
        updates["evidence_top_n"] = result.get("top_k", EXPAND_TOP_K)
        _nb = result.get("neighbors") or []
        if _nb:
            updates["neighbors"] = _nb
        _raw = result.get("top1_distance")
        if _raw is not None:
            updates["top1_distance"] = float(_raw)
        # ★ 재검색 결과 → PatchCore nd (판정 nd)
        _apply_patchcore_after_research(updates, _nb or state.get("neighbors") or [],
                                        _raw if _raw is not None else state.get("top1_distance"))
        # ★ [V3] fetch_contrast 보조 — wm_axes 첫 축의 대조 증거 1건 (판정 불개입,
        #   trace/감사 주석 전용). retriever field_filters 신규 경로 사용.
        if RESEARCH_MODE == "v3":
            _txn  = state.get("transaction") or {}
            _axes = (state.get("analyzer") or {}).get("wm_axes") or []
            _ax = _tv = None
            for _cand in _axes:                       # ★ 첫 축이 안 풀리면 다음 축으로
                _k = _resolve_axis(_cand, _txn)
                if _k is not None and _txn.get(_k) is not None:
                    _ax, _tv = _k, _txn[_k]
                    break
            if _ax is None and _axes:
                logger.warning("contrast_axis_unresolved", axes=_axes[:3],
                               txn_keys=list(_txn)[:12])
            if _ax and _tv is not None:
                try:
                    async with httpx.AsyncClient(timeout=5.0) as _c2:
                        _r2 = await _c2.post(f"{RETRIEVER_URL}/retrieve", json={
                            "embedding": state.get("embedding"),
                            "cluster_ids": [state.get("cluster_id", -1)],
                            "vec_index": state.get("vec_index"),
                            "top_k": 1, "field_filters": {_ax: str(_tv)}})
                        _r2.raise_for_status()
                        _d2 = _r2.json()
                    updates["_contrast_axis"] = _ax
                    updates["_contrast_distance"] = _d2.get("top1_distance")
                    _cn = (_d2.get("results") or [None])[0]
                    updates["_contrast_neighbor"] = _cn
                    updates["history"].append(f"fetch_contrast:{_ax}={_tv}")
                except Exception as _e2:
                    logger.warning("fetch_contrast_failed", error=str(_e2))
    except Exception as e:
        logger.warning("expand_tool_failed", error=str(e), mode=TOOL_EXECUTION_MODE)
        updates["evidence_top_n"] = EXPAND_TOP_K
        # ★ [OBS] 예외를 삼키면 nd 불변이 "재검색이 효과 없음"으로 오독된다.
        updates["_research_error"] = f"expand:{type(e).__name__}:{str(e)[:120]}"
        updates["_patchcore_status"] = "tool_failed"
    updates["_research_ms"] = round((time.time() - _t0) * 1000, 1)   # ★ round1 재검색 시간
    return updates


@instrument_node("reroute", required_keys=("cluster_id", "transaction"))
async def node_reroute(state: AgentState) -> AgentState:
    updates: Dict = {
        "expansion_used": True,
        "round": state.get("round", 0) + 1,
        "history": [f"reroute_cross_cluster:{TOOL_EXECUTION_MODE}"],
    }
    _t0 = time.time()
    try:
        result = await _call_tool("reroute_cross_cluster", state)
        if RESEARCH_MODE == "v3" and result.get("global"):
            # ★ [V3] verify_global 트레이스 — nd_global/flip은 별도 컬럼으로만
            #   기록해 legacy 지표와 혼합되지 않게 한다 (score_column 원칙).
            _rawg = result.get("top1_distance")
            if _rawg is not None:
                _ndg = _normalize_dist_fallback(float(_rawg))
                _nd_prev = (state.get("nd_history") or [None])[-1]
                updates["_nd_global"] = round(_ndg, 4)
                updates["_flip"] = bool(_nd_prev is not None
                                        and BOUNDARY_LOW <= float(_nd_prev) < BOUNDARY_HIGH
                                        and _ndg < BOUNDARY_LOW)
            updates["_global_cluster"] = result.get("cluster_id")
            updates["history"].append(
                f"verify_global:cluster_{state.get('cluster_id', -1)}"
                f"->global_top1_{result.get('cluster_id')}")
            # persona는 배정 cluster 것을 유지 — cluster는 설명용, 점수는 global
        elif result.get("cluster_id") is not None:
            updates["cluster_id"] = result["cluster_id"]
            # ★ [persona 교체] top2 후보 persona로 전환 — analyzer가 새 cluster
            # 기준 z-score 블록을 받는다(evaluator-once면 미사용이나 정합 유지).
            _cand = _pick_reroute_candidate(state)
            _new_persona = (_cand or {}).get("persona")
            updates["persona"] = _new_persona
            updates["cluster_persona_summary"] = (_new_persona or {}).get("description")
            updates["history"].append(
                f"reroute:cluster_{state.get('cluster_id', -1)}->{result['cluster_id']}")
        _nb = result.get("neighbors") or []
        if _nb:
            updates["neighbors"] = _nb
        _raw = result.get("top1_distance")
        if _raw is not None:
            updates["top1_distance"] = float(_raw)
        updates["evidence_top_n"] = result.get("top_k", BASE_TOP_K)
        # ★ 새 cluster 재검색 결과 → PatchCore nd (판정 nd)
        _apply_patchcore_after_research(updates, _nb or state.get("neighbors") or [],
                                        _raw if _raw is not None else state.get("top1_distance"))
    except Exception as e:
        logger.warning("reroute_tool_failed", error=str(e), mode=TOOL_EXECUTION_MODE)
        updates["_research_error"] = f"reroute:{type(e).__name__}:{str(e)[:120]}"
        updates["_patchcore_status"] = "tool_failed"
    updates["_research_ms"] = round((time.time() - _t0) * 1000, 1)   # ★ round1 재검색 시간
    return updates


@instrument_node("finalize")
async def node_finalize(state: AgentState) -> AgentState:
    # ★ [Observability ③] 종료 사유 기록 — route_after_plan은 순수 라우터라
    # state를 못 바꾸므로, finalize에서 동일 조건을 재판정해 기록한다.
    if state.get("track") == "fast":
        reason = "fast_track"
    elif state.get("action") == "accept_verdict":
        reason = "accept_verdict"
    elif state.get("action") == "stop":
        reason = "planner_stop"                    # ★ [Planner Serving]
    elif state.get("action") in ("global_knn", "contextual_knn"):
        reason = "planner_research_done"
    elif state.get("round", 0) >= state.get("budget", AGENT_MAX_ROUNDS) \
            and state.get("action") in ("expand_within_cluster", "reroute_cross_cluster"):
        reason = "budget_exceeded"
    else:
        reason = "unknown_action"

    # ★ [재설계 — min-fusion 폐기] 판정 nd 단일 선택:
    #   · 재검색함(expand/reroute) → 마지막 nd = PatchCore top-5 nd (nd_history[-1])
    #   · 재검색 안 함            → round0 raw top-1 nd (nd_history[0])
    #   두 경우 모두 nd_history의 마지막 원소가 판정 nd다(재검색 노드가 append,
    #   confident-skip/analyze는 round0 nd를 append). min을 취하지 않는다 —
    #   척도가 섞이지 않도록. forward(모델 cls)와 달리 agent는 여기서 nd로 판정.
    upd: Dict = {"history": ["finalize"], "termination": reason}
    nds  = state.get("nd_history") or []
    if nds:
        nd_round0 = nds[0]
        nd_final  = nds[-1]                       # 재검색 시 patchcore nd, 아니면 round0 nd
        researched = bool(state.get("expansion_used"))
        upd["round_log"] = [{
            "round": "final",
            "nd_round0": round(nd_round0, 4),
            "nd_final":  round(nd_final, 4),
            "n_rounds":  len(nds),
            "researched": researched,
            "nd_source": state.get("_nd_source", "raw_top1"),
            "reweight_w": state.get("_last_reweight_w"),
            "reweight_mode": PATCHCORE_REWEIGHT_MODE if researched else None,
            # ★ [OBS] Δnd=0 원인 규명용 — harness/report 가 그대로 집계한다
            "nd_delta":         round(nd_final - nd_round0, 4),
            "patchcore_status": state.get("_patchcore_status",
                                          "not_researched" if not researched else "unknown"),
            "plateau":          state.get("_last_plateau"),
            "research_error":   state.get("_research_error"),
        }]
        upd["verdict"] = "ABNORMAL" if nd_final >= CLS_ND_THRESHOLD else "NORMAL"
        _raws = state.get("raw_dist_history") or []
        if _raws:
            upd["top1_distance"] = _raws[-1]      # 응답 표시 거리도 판정 frame 정합
        if researched:
            logger.info("agent_research_verdict",
                        request_id=state.get("request_id", "?"),
                        nd_round0=round(nd_round0, 4), nd_final=round(nd_final, 4),
                        nd_source=state.get("_nd_source"), verdict=upd["verdict"])
    return upd


def route_after_pre_router(state: AgentState) -> str:
    return "finalize" if state.get("track") == "fast" else "retrieve"


def route_after_plan(state: AgentState) -> str:
    action = state.get("action", "accept_verdict")
    # 종료 조건: accept_verdict OR round>=budget OR expansion_used(이미 한번 돈 경우)
    # ★ [Planner Serving] GRPO action space는 legacy enum과 별개 공간이다.
    #   stop -> 즉시 종료 / 재검색 계열 -> node_research (budget 가드 동일).
    if action == "stop":
        return "finalize"
    if action in planner_action_names() and action != "stop":
        # ★ [1-step 고정] 이미 재검색했거나 step 상한에 도달하면 종료.
        #   budget(AGENT_MAX_ROUNDS)와 별개로 PLANNER_MAX_STEPS가 상한이다 —
        #   budget을 올려도 planner 경로는 1-step을 넘지 않는다.
        if state.get("expansion_used") or state.get("round", 0) >= PLANNER_MAX_STEPS:
            return "finalize"
        if state.get("round", 0) >= state.get("budget", AGENT_MAX_ROUNDS):
            agent_loop_budget_exceeded_total.inc()
            return "finalize"
        return "research"
    if action == "accept_verdict":
        return "finalize"
    if state.get("round", 0) >= state.get("budget", AGENT_MAX_ROUNDS):
        agent_loop_budget_exceeded_total.inc()
        return "finalize"
    if action == "expand_within_cluster":
        return "expand"
    if action == "reroute_cross_cluster":
        return "reroute"
    return "finalize"  # compare_features 등 미구현 action은 안전하게 종료

def _path_signature(trace: List[str]) -> str:
    """history -> Grafana group-by용 축약 문자열. 예: 'deep+expand' / 'fast+none'."""
    is_fast = "retrieve" not in trace
    tool = ("expand" if any(h.startswith("expand_within_cluster") for h in trace)
            else "reroute" if any(h.startswith("reroute_cross_cluster") for h in trace)
            else "none")
    return f"{'fast' if is_fast else 'deep'}+{tool}"
def build_agent_graph():
    if not _LANGGRAPH_OK:
        return None
    graph = StateGraph(AgentState)
    graph.add_node("pre_router", node_pre_router)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("analyze", node_analyze)
    graph.add_node("plan", node_plan)
    graph.add_node("expand", node_expand)
    graph.add_node("reroute", node_reroute)
    graph.add_node("research", node_research)     # ★ [Planner Serving] 신규 단일 노드
    graph.add_node("finalize", node_finalize)

    graph.set_entry_point("pre_router")
    graph.add_conditional_edges("pre_router", route_after_pre_router,
                                {"retrieve": "retrieve", "finalize": "finalize"})
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", "plan")
    # graph.add_conditional_edges("plan", route_after_plan, {
    #     "expand": "expand", "reroute": "reroute", "finalize": "finalize",
    # })
    graph.add_conditional_edges("plan", route_after_plan, {
        "expand": "expand", "reroute": "reroute",
        "research": "research",                    # ★ [Planner Serving]
        "finalize": "finalize",
    })
    # tool 실행 후 다시 analyze로 (loop) — re-retrieve 후 sufficiency 재평가
    graph.add_edge("expand", "analyze")
    graph.add_edge("reroute", "analyze")
    graph.add_edge("research", "analyze")   # ★ EVALUATOR_ONCE=true면 analyze가
                                        #   LLM 없이 confident를 반환 -> plan에서
                                        #   round>=1 가드로 stop -> finalize.
    graph.add_edge("finalize", END)
    return graph.compile()


agent_graph = build_agent_graph()


# ============================================================
# FastAPI endpoints
# ============================================================

@app.post("/plan", response_model=PlanResponse)
async def plan(req: PlanRequest):
    try:
        result = await run_plan(req)
        logger.info("plan_done", request_id=result.request_id, action=result.action,
                    backend=result.backend, latency_ms=result.latency_ms)
        return result
    except Exception as e:
        logger.error("planner_validation_failed", error=str(e))
        return fallback_plan(req, reason="validation_error")


@app.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    """
    Agent loop entrypoint — orchestrator_agentic.py 의 /detect 가 deep
    track 판정 후 호출하는 단일 진입점 (router+retriever base call 결과를
    base_* 필드로 동봉). base_* 가 비어 있으면(=controller 단독 호출,
    예: batch_inference_agent.py 의 스모크 테스트) controller가 자체적으로
    pre_router부터 다시 수행한다.
    LangGraph StateGraph(pre_router→retrieve→analyze→plan→tool→loop)를 실행.
    """
    if agent_graph is None:
        return DetectResponse(
            request_id=req.request_id or str(uuid.uuid4())[:8],
            track="deep", classification="NORMAL",
            reasoning="langgraph_not_available", planner_used=False,
            agent_trace=["error:langgraph_missing"],
            tool_execution_mode=TOOL_EXECUTION_MODE,
        )

    start = time.time()
    request_id = req.request_id or str(uuid.uuid4())[:8]
    base_provided = req.base_cluster_id is not None
    init_state: AgentState = {
        "request_id": request_id,
        "transaction": req.transaction,
        "embedding": req.embedding,
        "test_idx": req.test_idx,
        "_global_distance": req.global_distance,  # type: ignore
        "_base_provided": base_provided,  # type: ignore
        "_base_cluster_id": req.base_cluster_id,  # type: ignore
        "_base_top1_distance": req.base_top1_distance,  # type: ignore
        "_base_distance_band": req.base_distance_band,  # type: ignore
        "_base_cluster_persona_summary": req.base_cluster_persona_summary,  # type: ignore
        "_base_neighbors": req.base_neighbors or [],  # type: ignore
        "vec_index": req.base_vec_index,               # type: ignore
        "top_k_clusters": req.base_top_k_clusters or [],  # type: ignore
        "persona": req.base_persona,                   # type: ignore
        "match_type": req.base_match_type,             # type: ignore
        "evidence_top_n": BASE_TOP_K,
        "neighbors": [],
        "round": 0,
        "budget": AGENT_MAX_ROUNDS,
        "expansion_used": False,
        "history": [],
        "analyzer_history": [],
        "round_log": [],
        "nd_history": [],
        "raw_dist_history": [],
    }
    try:
        final_state = await agent_graph.ainvoke(init_state)
        latency_ms = (time.time() - start) * 1000
        analyzer_out = final_state.get("analyzer", {})
        # analyzer_out은 node_analyze에서 raw(analyzer 원본 응답)로 저장됨.
        # DetectResponse 조립을 위해 normalize를 한 번 거쳐 표준 필드를 얻는다.
        # (node_plan의 model_validator와 달리 여기는 PlanRequest를 거치지
        #  않으므로 직접 normalize_analyzer_output을 호출한다.)
        normalized_for_response = normalize_analyzer_output(analyzer_out)
        classification = final_state.get("verdict") or normalized_for_response.classification
        trace = final_state.get("history", [])
        agent_path_total.labels(path=_path_signature(trace)).inc()
        agent_rounds_histogram.observe(final_state.get("round", 0))
        planner_used = any(h.startswith("plan:") for h in trace)        
        detect_requests_total.labels(status="success", track=final_state.get("track", "deep")).inc()
        detect_latency.observe(latency_ms / 1000)
        return DetectResponse(
            request_id=request_id,
            track=final_state.get("track", "deep"),
            classification=classification,
            confidence=normalized_for_response.confidence,
            top_1_distance=final_state.get("top1_distance"),
            primary_cluster_id=final_state.get("cluster_id"),
            reasoning=normalized_for_response.reasoning_text or "no_reasoning",
            planner_used=planner_used,
            agent_trace=trace,
            latency_ms=latency_ms,
            tool_execution_mode=TOOL_EXECUTION_MODE,
            agent_rounds=final_state.get("round_log", []),
            termination=final_state.get("termination"),
        )
    except Exception as e:
        logger.error("detect_failed", error=str(e))
        detect_requests_total.labels(status="error", track="unknown").inc()
        return DetectResponse(
            request_id=request_id, track="deep", classification="NORMAL",
            reasoning=f"agent_loop_error: {e}", planner_used=False,
            agent_trace=["error"], latency_ms=(time.time() - start) * 1000,
            tool_execution_mode=TOOL_EXECUTION_MODE,
        )


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "planner", "version": "4.0.0-langgraph",
            "wm_schema":      WM_SCHEMA,
            "research_mode":  RESEARCH_MODE,
            # "planner_backend": PLANNER_BACKEND,
            "tool_execution_mode": TOOL_EXECUTION_MODE,
            }


@app.get("/ready")
async def ready():
    return {"status": "ready", "model": LLM_MODEL,
            "vertex_ready": vertexAI is not None,
            "langgraph_ready": agent_graph is not None,
            "actions": list(FOLLOWUP_ACTIONS)}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.on_event("startup")
async def startup_event():
    logger.info("planner_started", model=LLM_MODEL, actions=list(FOLLOWUP_ACTIONS),
                langgraph_ready=agent_graph is not None)

# 손수 주석한 정본 그래프 — draw_mermaid() 자동 출력은 노드명만 나와
# 이 프로젝트의 핵심(🟠 LLM은 evaluator 1회만 / 🔵 판정·라우팅은 코드)이
# 안 보이므로 포트폴리오용으로 별도 관리한다. system architecture와 1:1 대응.
ANNOTATED_MERMAID = """flowchart TD
    START([Test Transaction]) --> PR{{"pre_router: fast/deep track"}}
    PR -->|"fast track"| FIN
    PR -->|"deep track"| RET["retrieve<br/>Filtered HNSW / evidence top-3"]
    RET --> ANA["analyze : Evaluator LLM<br/>what_missing 1-key 생성<br/>cls/suf = 코드(nd) 결정론"]
    ANA --> PLAN{{"plan : Planner<br/>what_missing to tool"}}
    PLAN -->|"confident"| FIN
    PLAN -->|"sparse-evidence"| EX["expand_within_cluster<br/>같은 cluster 재검색"]
    PLAN -->|"cluster mismatch"| RR["reroute_cross_cluster<br/>rank2 cluster + persona 교체"]
    EX --> PC["재검색 nd 재계산<br/>PatchCore top-5 / raw"]
    RR --> PC
    PC -.->|"evaluator 재호출 없음"| FIN
    FIN["finalize<br/>재검색O: patchcore nd / X: raw nd<br/>nd >= threshold : ABNORMAL"] --> DONE([API Response])

    classDef llm fill:#F58518,stroke:#b5610f,color:#fff;
    classDef code fill:#4C78A8,stroke:#2f4b6e,color:#fff;
    classDef term fill:#54A24B,stroke:#356,color:#fff;
    class ANA llm;
    class PR,PLAN,RET,EX,RR,PC,FIN code;
    class START,DONE term;
"""


@app.get("/graph/mermaid")
async def graph_mermaid(annotated: bool = True):
    """Mermaid 텍스트 반환. annotated=True(기본): 역할분리·evaluator-once·
    PatchCore nd를 주석한 정본. annotated=false: LangGraph 자동생성(구조검증용)."""
    if not annotated:
        if agent_graph is None:
            raise HTTPException(status_code=503, detail="graph not built")
        return {"mermaid": agent_graph.get_graph().draw_mermaid(), "annotated": False}
    return {"mermaid": ANNOTATED_MERMAID, "annotated": True}

@app.get("/graph/png")
async def graph_png():
    """PNG 이미지로 반환 (graphviz 필요)"""
    import io
    from fastapi.responses import Response
    png_bytes = agent_graph.get_graph().draw_mermaid_png()
    return Response(content=png_bytes, media_type="image/png")

# ============================================================
# ★ [GRPO Planner] Oracle Rollout — /rollout
# ============================================================
# 목적: state 1개당 GRPO action space(stop/global_knn/contextual_knn×feature
# 후보)를 전수 실행해 각 action의 결과(nd_after, Δmargin, cost, support,
# neighbor identity)를 반환한다. 오라클(§8 "counterfactual reference")
# 계산에 필요한 원재료를 만드는 지점 — 오라클 집계(argmax 산출, V1~V3 검증,
# parquet 저장)는 이 엔드포인트를 반복 호출하는 별도 배치 job(hard_negative_
# miner.py/agent_batch_inference.py와 동일 패턴)에서 다음 턴에 진행한다.
#
# ★ 기존 /detect(LangGraph 루프)와 완전히 분리된 별도 경로다. LangGraph는
# "하나의 action을 고르고 state를 전진시키는" 순차 실행 구조라 여러 action을
# 같은 state에서 나란히 실행하는 rollout에 맞지 않는다 — 그래서 node_pre_
# router/node_retrieve/node_analyze(round 0)까지는 재사용하되, plan 이후는
# LangGraph를 타지 않고 이 함수가 직접 후보를 순회한다. 그래프/기존 action
# space(FOLLOWUP_ACTIONS)/route_after_plan은 전혀 건드리지 않는다.
#
# ★ [범위 제한 — 이번 turn] TOOL_EXECUTION_MODE와 무관하게 mcp_native 경로만
# 구현한다(디버깅 편의 우선순위 요청). direct/mcp(REST shim) 대응 rollout
# 실행기는 다음 단계에서 필요 시 추가.

_GRPO_MCP_CLIENT_URL = TOOLS_MCP_URL  # agent_tools_mcp_server 재사용 (동일 서비스)

# ★ MCP 세션 재사용 + 동시성 제한.
#   호출마다 MultiServerMCPClient를 만들면 44개 후보가 각자 streamablehttp
#   세션(anyio TaskGroup)을 열고, asyncio.gather의 서로 다른 task scope에서
#   취소되며 "unhandled errors in a TaskGroup"이 난다. tools를 1회만 받아
#   캐시하고, 동시 invoke 수를 제한한다.
_GRPO_TOOL_CACHE: Dict[str, Any] = {}
_GRPO_TOOL_LOCK = asyncio.Lock()
_GRPO_MCP_SEM = asyncio.Semaphore(int(os.getenv("GRPO_MCP_CONCURRENCY", "16")))

# ★ [오라클 경로 최적화] MCP를 우회하고 retriever를 직접 호출한다.
#   langchain_mcp_adapters는 tool.ainvoke()마다 streamable_http 세션을
#   새로 연다(TCP + MCP initialize + teardown). 실측 후보당 4.2초로,
#   ES 검색(20~40ms)의 100배다. 오라클 수집에 MCP 계층은 필요 없으므로
#   직접 호출로 바꾼다. 서빙 경로(node_plan → MCP)는 그대로 둔다.
ROLLOUT_DIRECT_RETRIEVER = os.getenv("ROLLOUT_DIRECT_RETRIEVER", "true").lower() == "true"
_DIRECT_SEM = asyncio.Semaphore(int(os.getenv("ROLLOUT_DIRECT_CONCURRENCY", "48")))
_DIRECT_CLIENT: Optional[httpx.AsyncClient] = None


def _direct_client() -> httpx.AsyncClient:
    """★ 커넥션 풀 재사용 — 후보마다 새 클라이언트를 만들면 TCP 핸드셰이크가
    반복된다."""
    global _DIRECT_CLIENT
    if _DIRECT_CLIENT is None or _DIRECT_CLIENT.is_closed:
        _n = int(os.getenv("ROLLOUT_DIRECT_CONCURRENCY", "48"))
        _DIRECT_CLIENT = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=_n * 2,
                                max_keepalive_connections=_n * 2),
            timeout=httpx.Timeout(float(os.getenv("ROLLOUT_DIRECT_TIMEOUT", "60"))))
    return _DIRECT_CLIENT


async def _call_tool_direct(tool_name: str, payload: Dict) -> Dict:
    """MCP tool과 동일한 반환 계약을 유지하면서 retriever를 직접 호출한다.
    (agent_tools_mcp_server.py의 global_knn / contextual_knn / attribute_knn과
     같은 요청을 만든다 — 계약이 어긋나면 오라클이 조용히 틀린다)"""
    _tx  = _norm_keys(payload.get("transaction"))
    _fs  = payload.get("features") or []
    _k   = int(payload.get("top_k") or EXPAND_TOP_K)
    req: Dict[str, Any] = {
        "embedding":   payload.get("embedding"),
        "cluster_ids": [],                     # 세 tool 모두 전역 스코프
        "vec_index":   payload.get("vec_index"),
        "top_k":       _k,
    }
    if tool_name == "contextual_knn":
        _sb = {f: str(_tx[f]) for f in _fs if _tx.get(f) is not None}
        if not _sb:
            return {"top_k": _k, "view": "contextual", "features_used": [],
                    "cluster_id": None, "top1_distance": None, "neighbors": [],
                    "support": 0, "_degraded": True, "_error": "no_features"}
        req["should_boost"] = _sb
    elif tool_name == "attribute_knn":
        _ff = {f: str(_tx[f]) for f in _fs if _tx.get(f) is not None}
        if not _ff:
            return {"top_k": _k, "view": "attribute", "features_used": [],
                    "cluster_id": None, "top1_distance": None, "neighbors": [],
                    "support": 0, "_degraded": True, "_error": "no_features"}
        req["field_filters"]  = _ff
        req["attribute_mode"] = True
        req["return_support"] = True
    elif tool_name != "global_knn":
        raise RuntimeError(f"direct 경로 미지원 tool: {tool_name}")

    async with _DIRECT_SEM:
        r = await _direct_client().post(f"{RETRIEVER_URL}/retrieve", json=req)
        r.raise_for_status()
        data = r.json()

    res  = data.get("results") or data.get("neighbors") or []
    _sup = data.get("support")
    _sup = int(_sup) if _sup is not None else len(res)
    return {
        "top_k": _k,
        "view": {"global_knn": "global", "contextual_knn": "contextual",
                 "attribute_knn": "attribute"}[tool_name],
        "features_used": list(req.get("field_filters") or req.get("should_boost") or {}),
        "cluster_id": (res[0].get("cluster_id") if res else None),
        "top1_distance": data.get("top1_distance"),
        "neighbors": res,
        "support": _sup,
        "_degraded": not res,
        "_error": None if res else "empty_result",
    }


def _unwrap_exc(e: BaseException) -> str:
    """ExceptionGroup의 sub-exception을 펼쳐 실제 원인을 보이게 한다."""
    subs = getattr(e, "exceptions", None)
    if subs:
        return " | ".join(_unwrap_exc(s) for s in subs)
    return f"{type(e).__name__}: {str(e)[:200]}"


async def _get_grpo_tools() -> Dict[str, Any]:
    if _GRPO_TOOL_CACHE:
        return _GRPO_TOOL_CACHE
    async with _GRPO_TOOL_LOCK:
        if _GRPO_TOOL_CACHE:
            return _GRPO_TOOL_CACHE
        client = MultiServerMCPClient({
            "agent_tools": {"url": f"{_GRPO_MCP_CLIENT_URL}/mcp",
                            "transport": "streamable_http"},
        })
        for t in await client.get_tools():
            _GRPO_TOOL_CACHE[t.name] = t
        logger.info("grpo_mcp_tools_cached", tools=sorted(_GRPO_TOOL_CACHE.keys()))
    return _GRPO_TOOL_CACHE

async def _call_grpo_tool_native(tool_name: str, payload: Dict) -> Dict:
    """
    GRPO action(global_knn/contextual_knn) 전용 MCP native 호출.
    _call_tool_mcp_native(legacy/v3 전용, expand/reroute/verify_global 계약)와
    별도 함수로 둔다 — payload 스키마가 다르고(features 등), legacy 경로의
    tool 스위칭 로직(RESEARCH_MODE 분기)을 여기 섞으면 두 planner의 실행
    경로가 꼬인다.
    """
    if not _MCP_NATIVE_OK:
        raise RuntimeError("langchain_mcp_adapters not installed — mcp_native mode unavailable")
    tools = await _get_grpo_tools()
    tool = tools.get(tool_name)
    if tool is None:
        raise RuntimeError(
            f"GRPO tool '{tool_name}' not found (discovered: {sorted(tools.keys())})")
    async with _GRPO_MCP_SEM:
        try:
            raw = await tool.ainvoke(payload)
        except BaseException as e:
            # ★ ExceptionGroup을 펼쳐 실제 원인을 남긴다
            raise RuntimeError(f"mcp_invoke_failed({tool_name}): {_unwrap_exc(e)}") from e
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
        return json.loads(raw[0]["text"])
    if isinstance(raw, str):
        return json.loads(raw)
    return raw
# =====================================================================
# controller.py — GRPO / Oracle rollout 섹션 교체본
# ---------------------------------------------------------------------
# 적용 방법:
#   기존 controller.py에서
#       `def _margin(nd: float, y: int, ...)` 정의 줄부터
#       파일 끝의 `if __name__ == "__main__":` 직전까지
#   를 이 파일의 내용으로 통째로 교체한다.
#   (그 위의 `_call_grpo_tool_native`는 그대로 둔다)
#
# 이번 개정의 핵심 4가지
#   1. attribute_knn 추가 — exact term filter 기반 sparse 축.
#      contextual(dense pool 재정렬)과 달리 부분집합/상위집합 관계를 깨므로
#      nd가 양방향으로 움직인다. cluster_mismatch의 정합 처방.
#   2. 2-step rollout — step1 ∈ {stop, contextual(f), attribute(f)},
#      step2는 step1과 "다른 tool"만. 같은 tool 반복은 같은 pool 재정렬이라
#      정보가 늘지 않으므로 금지(조합 폭발 억제 효과도 있음).
#   3. reward를 저장하지 않고 원자적 관측만 저장 — nd/이웃 identity/이웃
#      feature/이웃 cluster_id/spread를 남기면 R_margin·R_flip·R_resolve·
#      R_auroc·R_diag를 전부 노트북에서 사후 계산할 수 있다(λ 사후계산과
#      동일한 원칙). reward 축이 바뀌어도 재수집이 필요 없다.
#   4. analyzer는 round0에서 딱 1회 — step1/step2 후보마다 부르지 않는다.
#      진단 해소 지표(Δfit/Δagree/Δspread)는 prompts.compute_evidence_axes()로
#      LLM 없이 결정론적으로 재계산한다. 오라클 수집 시간의 대부분이 여기서
#      절약된다.
# =====================================================================

# ---------------------------------------------------------------------
# 상수 (파일 상단 설정 블록으로 옮겨도 됨 — 여기 둔 것은 diff 가독성 때문)
# ---------------------------------------------------------------------
# cost는 '검색 호출 수'로 통일한다. 실측 latency(ms)를 reward에 넣으면
# 인덱스 augmentation 여부/contextual 재구현/tool 추가마다 학습 설정이
# 무효화된다. latency는 SLA 검증 지표로만 별도 기록한다.
ROLLOUT_ACTION_COST = {
    "stop":           0.0,
    "contextual_knn": 1.0,
    "attribute_knn":  1.0,
    "global_knn":     1.0,
}
# ★ [1-step 고정] 기본값을 PLANNER_MAX_STEPS(서빙 상한)에서 승계한다 —
#   오라클과 서빙의 step 수가 갈라지지 않게 하는 단일 출처.
ROLLOUT_MAX_STEPS = int(os.getenv("ROLLOUT_MAX_STEPS", str(PLANNER_MAX_STEPS)))
ROLLOUT_STEP2_TOPN  = int(os.getenv("ROLLOUT_STEP2_TOPN", "3"))   # step1 상위 N개에서만 분기
ROLLOUT_MAX_FEATS   = int(os.getenv("ROLLOUT_MAX_FEATS", "2"))    # 조합 크기 상한
ROLLOUT_TOOLS       = [t.strip() for t in
                       os.getenv("ROLLOUT_TOOLS", "contextual_knn,attribute_knn").split(",")
                       if t.strip()]
# global_knn은 참고용 — ROLLOUT_TOOLS에 넣지 않아도 아래 스위치로 1개만 추가 실행.
ROLLOUT_INCLUDE_GLOBAL = os.getenv("ROLLOUT_INCLUDE_GLOBAL", "true").lower() == "true"
# 오라클 수집 시 후보마다 analyzer를 부르지 않는다(기본 true). 서빙 /detect는 무관.
SKIP_ANALYZER_IN_ROLLOUT = os.getenv("SKIP_ANALYZER_IN_ROLLOUT", "true").lower() == "true"

_CATEGORICAL_FALLBACK = ["sex", "source", "browser",
                         "IP_country", "month_purchase", "weekday_purchase"]


# ---------------------------------------------------------------------
# 보조 함수
# ---------------------------------------------------------------------
def _margin(nd: float, y: int, tau: float = CLS_ND_THRESHOLD) -> float:
    """margin(nd,y) = (2y-1)*(nd-tau) — 인수인계 §1 Reward 정의 그대로.
    ★ Δmargin은 tau가 소거되므로 tau 값에 불변이다(균등 커널). 반면
    flip/resolve 커널은 tau에 전적으로 의존하므로, 오라클에는 tau를 고정한
    파생값이 아니라 nd 원본을 남긴다."""
    return (2 * int(y) - 1) * (float(nd) - float(tau))


def _evidence_from_analyzer(state: Dict) -> Dict[str, Any]:
    """local_analyzer.py의 AnalyzeResponse가 반환한 evidence 축을 그대로 읽는다.
    mock/fallback 경로에서는 빈 값으로 안전하게 degrade한다."""
    a = state.get("analyzer") or {}
    return {
        "agrees":        a.get("agrees") or [],
        "differs":       a.get("differs") or [],
        "deviant":       a.get("deviant") or [],
        "rare":          a.get("rare") or [],
        "persona_match": a.get("persona_match") or [],
        "spread_tag":    a.get("spread_tag"),
    }


def _norm_keys(d: Optional[Dict]) -> Dict:
    """ES 문서 키 정규화 — aug 인덱스는 'IP country' 같은 공백 컬럼명으로
    색인돼 있다. 여기서 언더스코어로 통일해야 agrees 비교가 성립한다.
    (retriever 쪽에서도 동일 정규화를 하지만 이중 방어)"""
    if not isinstance(d, dict):
        return {}
    return {str(k).replace(" ", "_"): v for k, v in d.items()}


def _neighbor_view(neighbors: Optional[List[Dict]]) -> Dict[str, Any]:
    """이웃 리스트에서 오라클이 남겨야 할 원자적 관측을 추출한다.
    features/cluster_id를 남겨야 R_diag(Δfit/Δagree/escape)를 재수집 없이
    사후 계산할 수 있다 — 지금까지 neighbor_oidx만 남겨서 불가능했던 부분."""
    neighbors = neighbors or []
    oidx, feats, cids, dists = [], [], [], []
    for n in neighbors:
        if not isinstance(n, dict):
            continue
        nn = _norm_keys(n)
        if nn.get("original_index") is not None:
            oidx.append(int(nn["original_index"]))
        _f = nn.get("features")
        feats.append(_norm_keys(_f) if isinstance(_f, dict)
                     else {k: v for k, v in nn.items()
                           if k not in ("original_index", "cluster_id", "distance", "score")})
        if nn.get("cluster_id") is not None:
            cids.append(int(nn["cluster_id"]))
        if nn.get("distance") is not None:
            dists.append(float(nn["distance"]))
    return {"oidx": oidx, "features": feats, "cluster_ids": cids, "distances": dists}


def _spread_of(distances: Optional[List[float]]) -> Optional[float]:
    """이웃 거리의 분산 지표 — neighbor_insufficient 해소 여부(Δspread)의 원재료.
    prompts의 spread 정의(표준편차)와 동일하게 맞춘다."""
    ds = [float(d) for d in (distances or []) if d is not None]
    if len(ds) < 2:
        return None
    m = sum(ds) / len(ds)
    return float((sum((d - m) ** 2 for d in ds) / len(ds)) ** 0.5)


def _evidence_after(state: Dict, nview: Dict) -> Dict[str, Any]:
    """★ analyzer(LLM) 호출 없이 결정론적으로 evidence 축을 재계산한다.
    prompts.compute_evidence_axes는 순수 함수이므로 후보 실행 결과만 있으면
    된다 — 후보마다 0.5B analyzer를 부르면 수집 시간이 수십 배가 된다."""
    try:
        from prompts import compute_evidence_axes  # type: ignore
    except Exception as e:
        # ★ 조용히 {}를 반환하면 agrees/differs/deviant/rare가 전부 비어
        #   wm_type이 무조건 accept로 떨어진다. 반드시 드러낸다.
        if not globals().get("_EVIDENCE_IMPORT_WARNED"):
            globals()["_EVIDENCE_IMPORT_WARNED"] = True
            logger.error("evidence_axes_import_failed",
                         error=f"{type(e).__name__}: {e}",
                         hint="controller 컨테이너에 prompts.py가 마운트되었는지 확인")
        return {}
    try:
        _ev = compute_evidence_axes(
            test_features=_norm_keys(state.get("transaction")),
            persona=state.get("persona"),
            distances=nview.get("distances") or None,
            topk_feats=nview.get("features") or None,
        ) or {}
        if not any(_ev.get(k) for k in ("agrees", "differs", "deviant",
                                        "rare", "persona_match")):
            logger.warning("evidence_after_empty",
                           n_tx=len(_norm_keys(state.get("transaction"))),
                           n_neigh=len(nview.get("features") or []),
                           n_dist=len(nview.get("distances") or []),
                           has_persona=bool(state.get("persona")),
                           persona_keys=sorted((state.get("persona") or {}).keys()))
        return _ev
    except Exception as e:  # 계산 실패는 수집 실패가 아니다 — 빈 dict로 degrade
        logger.warning("evidence_after_failed", error=f"{type(e).__name__}: {e}")
        return {}


def _feature_combos(pool: List[str], max_k: int = None) -> List[List[str]]:
    """candidate feature 조합 — 1개 조합 + 2개 조합(순서 무관, 중복 제거)."""
    max_k = max_k or ROLLOUT_MAX_FEATS
    pool = list(dict.fromkeys([p for p in (pool or []) if p]))
    out: List[List[str]] = []
    for k in range(1, max_k + 1):
        out.extend([list(c) for c in itertools.combinations(pool, k)])
    return out


def _build_step_space(pool: List[str], exclude_tool: Optional[str] = None) -> List[Dict]:
    """한 step의 action space. exclude_tool이 주어지면 그 tool은 제외한다
    (2-step에서 같은 tool 반복 금지 — 같은 pool 재정렬이라 정보가 안 늘어난다)."""
    space: List[Dict] = [{"action": "stop", "features": []}]
    combos = _feature_combos(pool)
    for tool in ROLLOUT_TOOLS:
        if tool == exclude_tool:
            continue
        for fs in combos:
            space.append({"action": tool, "features": fs})
    if ROLLOUT_INCLUDE_GLOBAL and exclude_tool != "global_knn":
        # 참고용 기준선 — feature 없음. 부분집합 정리상 nd의 하한을 준다.
        space.append({"action": "global_knn", "features": []})
    return space


# ---------------------------------------------------------------------
# 후보 1개 실행
# ---------------------------------------------------------------------
async def _execute_candidate(cand: Dict, state: Dict, y: int, nd_before: float,
                             *, base_nview: Optional[Dict] = None) -> Dict:
    """
    action 1개를 실행하고 원자적 관측을 반환한다. state는 읽기 전용
    (후보들이 같은 base state에서 독립적으로 갈라져야 하므로 mutate 금지).

    ★ reward를 계산해 넣지 않는다 — nd/이웃/spread/evidence만 남기고
    R_margin·R_flip·R_resolve·R_auroc·R_diag는 전부 노트북에서 파생시킨다.
    (호환을 위해 delta_margin/cost/reward는 참고값으로 함께 넣되, 학습은
     원자적 관측에서 재계산한 값을 쓴다)
    """
    action = cand["action"]
    feats_req = list(cand.get("features", []) or [])
    t0 = time.time()

    if action == "stop":
        nd_after       = nd_before
        nview          = dict(base_nview or {"oidx": [], "features": [], "cluster_ids": [], "distances": []})
        support        = None
        features_used  = []
        degraded       = False
        error          = None
    else:
        payload: Dict[str, Any] = {
            "embedding": state.get("embedding"),
            "vec_index": state.get("vec_index"),
            "top_k": EXPAND_TOP_K,
        }
        if action in ("contextual_knn", "attribute_knn"):
            _tx = _norm_keys(state.get("transaction"))
            payload["features"] = feats_req
            payload["transaction"] = _tx
            # if action == "attribute_knn":
            #     # attribute는 '값 일치'가 필요하므로 feature명이 아니라 name->value
            #     payload["features"] = {f: _tx.get(f) for f in feats_req if _tx.get(f) is not None}
        try:
            result = (await _call_tool_direct(action, payload)
                      if ROLLOUT_DIRECT_RETRIEVER
                      else await _call_grpo_tool_native(action, payload))
            raw_top1 = result.get("top1_distance")
            nd_after = (_normalize_dist_fallback(float(raw_top1))
                        if raw_top1 is not None else nd_before)
            nview         = _neighbor_view(result.get("neighbors"))
            support       = int(result.get("support") if result.get("support") is not None
                                else len(nview["oidx"]))
            features_used = result.get("features_used", feats_req)
            degraded      = bool(result.get("_degraded") or result.get("degraded"))
            error         = result.get("_error") if result.get("_error") else None
            # ★ support=0/None은 실행 실패가 아니라 "이 조합으로는 결과가 없다"는
            #   유효한 관측이다. error로 기록하면 실패율이 폭증하고, planner가
            #   피해야 할 조합을 배울 근거가 사라진다.
            if degraded and not (nview.get("oidx") or []):
                error = None
                nd_after = nd_before
        except Exception as e:
            logger.warning("grpo_candidate_failed", action=action,
                           features=feats_req, error=_unwrap_exc(e))
            nd_after      = nd_before
            nview         = {"oidx": [], "features": [], "cluster_ids": [], "distances": []}
            support       = None
            features_used = feats_req
            degraded      = True
            error         = _unwrap_exc(e)

    # ── 진단 해소 지표의 원재료 (LLM 호출 없음) ──
    ev_after   = _evidence_after(state, nview) if action != "stop" else {}
    base_cid   = state.get("cluster_id")
    _cids      = nview.get("cluster_ids") or []
    escape     = (round(sum(1 for c in _cids if c != base_cid) / len(_cids), 4)
                  if _cids and base_cid is not None else None)

    margin_before = _margin(nd_before, y)
    margin_after  = _margin(nd_after, y)
    cost          = ROLLOUT_ACTION_COST.get(action, 1.0)

    return {
        "action":         action,
        "features":       feats_req,
        "features_used":  features_used,
        # ── 원자적 관측 (여기서 모든 reward 커널이 파생된다) ──
        "nd_before":      round(float(nd_before), 6),
        "nd_after":       round(float(nd_after), 6),
        "nd_topk":        [round(d, 6) for d in (nview.get("distances") or [])],
        "spread_after":   _spread_of(nview.get("distances")),
        "neighbor_oidx":        nview.get("oidx") or [],
        "neighbor_cluster_ids": _cids,
        "neighbor_features":    nview.get("features") or [],
        "cross_cluster_escape": escape,
        # ── 결정론적 evidence 재계산 (Δagree/Δfit용) ──
        "agrees_after":        ev_after.get("agrees") or [],
        "differs_after":       ev_after.get("differs") or [],
        "deviant_after":       ev_after.get("deviant") or [],
        "rare_after":          ev_after.get("rare") or [],
        "persona_match_after": ev_after.get("persona_match") or [],
        "spread_tag_after":    ev_after.get("spread_tag"),
        # ── 참고값 (학습은 위 원자적 관측에서 재계산해 쓴다) ──
        "margin_before":  round(margin_before, 6),
        "margin_after":   round(margin_after, 6),
        "delta_margin":   round(margin_after - margin_before, 6),
        "cost":           cost,
        "support":        support,
        "degraded":       degraded,
        "latency_ms":     round((time.time() - t0) * 1000, 1),
        "error":          error,
    }


# ---------------------------------------------------------------------
# Rollout API
# ---------------------------------------------------------------------
class RolloutRequest(BaseModel):
    """/detect(DetectRequest)와 동일한 base_* 계약 재사용. 오라클 계산에는
    ground-truth label(y)이 필수라는 점만 다르다."""
    request_id:    Optional[str] = None
    transaction:   Dict = Field(default_factory=dict)
    embedding:     Optional[List[float]] = None
    test_idx:      Optional[int] = None
    y:             int = Field(..., description="ground-truth: 0=NORMAL, 1=ABNORMAL")

    base_cluster_id:              Optional[int] = None
    base_top1_distance:           Optional[float] = None
    base_distance_band:           Optional[str] = None
    base_cluster_persona_summary: Optional[str] = None
    base_vec_index:               Optional[str] = None
    base_match_type:              Optional[str] = None
    base_top_k_clusters:          Optional[List[Dict]] = None
    base_persona:                 Optional[Dict] = None
    base_neighbors:               Optional[List[Dict]] = None

    # ★ 수집 제어 (미지정 시 환경변수 기본값)
    max_steps:      Optional[int] = None
    step2_topn:     Optional[int] = None


class RolloutResponse(BaseModel):
    request_id:     str
    test_idx:       Optional[int] = None
    y:              int
    cluster_id:     Optional[int] = None
    wm_type:        Optional[str] = None
    sufficiency:    Optional[str] = None
    wm_axes:        Optional[List[str]] = None
    agrees:         List[str] = Field(default_factory=list)
    differs:        List[str] = Field(default_factory=list)
    deviant:        List[str] = Field(default_factory=list)
    rare:           List[str] = Field(default_factory=list)
    persona_match:  List[str] = Field(default_factory=list)
    spread_tag:     Optional[str] = None
    nd_before:      float
    # ★ round0 원재료 — planner 프롬프트 재구성 + Δfit/Δspread의 'before' 기준
    persona:        Optional[Dict] = None
    neighbors:      Optional[List[Dict]] = None
    base_spread:    Optional[float] = None
    base_neighbor_cluster_ids: List[int] = Field(default_factory=list)
    candidate_pool: List[str] = Field(default_factory=list)
    # ★ step1 후보 + step2 후보(부모 참조 포함)를 한 리스트에 담는다
    candidates:     List[Dict] = Field(default_factory=list)
    oracle_action:  Optional[str] = None   # 참고값(Δmargin 기준) — 집계 스크립트가 재계산
    n_steps:        int = 1
    latency_ms:     float = 0.0


@app.post("/rollout", response_model=RolloutResponse)
async def rollout(req: RolloutRequest):
    """
    state 1개 → (최대 2-step) action space 전수 실행 → 각 경로의 원자적 관측 반환.

    파이프라인:
      pre_router/retrieve(round0) → analyze(round0, LLM 1회)
      → step1 space = {stop} ∪ {tool(f) : tool ∈ ROLLOUT_TOOLS, |f| ≤ 2} (+global 1개)
      → step1 상위 N개에서 분기 → step2 space = {stop} ∪ {다른 tool(f)}
      → 각 후보 mcp_native 병렬 실행

    ★ analyzer는 round0 1회만. step1/step2 후보의 evidence는
      prompts.compute_evidence_axes로 결정론적 재계산(LLM 미호출).
    """
    start = time.time()
    request_id = req.request_id or str(uuid.uuid4())[:8]
    max_steps  = int(req.max_steps or ROLLOUT_MAX_STEPS)
    step2_topn = int(req.step2_topn or ROLLOUT_STEP2_TOPN)
    base_provided = req.base_cluster_id is not None

    state: Dict = {
        "request_id": request_id,
        "transaction": _norm_keys(req.transaction),
        "embedding": req.embedding,
        "test_idx": req.test_idx,
        "_base_provided": base_provided,
        "_base_cluster_id": req.base_cluster_id,
        "_base_top1_distance": req.base_top1_distance,
        "_base_distance_band": req.base_distance_band,
        "_base_cluster_persona_summary": req.base_cluster_persona_summary,
        "_base_neighbors": req.base_neighbors or [],
        "vec_index": req.base_vec_index,
        "top_k_clusters": req.base_top_k_clusters or [],
        "persona": req.base_persona,
        "match_type": req.base_match_type,
        "evidence_top_n": BASE_TOP_K,
        "neighbors": [],
        "round": 0,
    }
    try:
        state.update(await node_retrieve(state))   # type: ignore[arg-type]
        nd_before = (state.get("nd_history") or [None])[-1]
        if nd_before is None:
            nd_before = _normalize_dist_fallback(float(state.get("top1_distance", DIST_P50)))
        nd_before = float(nd_before)
        # ★ [SKIP_ANALYZER_IN_ROLLOUT] 오라클 수집에서 analyzer(0.5B, CPU)는
        #   state당 ~15초를 먹는데, 후보 실행에는 전혀 쓰이지 않는다.
        #   필요한 건 evidence 축(agrees/differs/deviant/rare/persona_match/
        #   spread_tag)과 wm_type뿐이고, 둘 다 prompts.compute_evidence_axes()로
        #   결정론적으로 유도된다 — analyzer의 나머지 출력(reasoning/cls 등)은
        #   오라클 계산에 미사용.
        if SKIP_ANALYZER_IN_ROLLOUT:
            _nv0 = _neighbor_view(state.get("neighbors"))
            _ev0 = _evidence_after(state, _nv0)
            _spr = _spread_of(_nv0.get("distances"))
            # ★ prompts.EVALUATOR_INSTRUCTION_V3 규칙 그대로:
            #   cluster_mismatch      : [DEVIANT] 또는 [RARE]가 하나라도 있음
            #   neighbor_insufficient : [spread] AND differs >= 2   (★ AND)
            #   accept                : 그 외
            #   (이전 `or len(agrees) < 2`는 V3에 없는 조건이라 dense한 state까지
            #    전부 neighbor_insufficient로 떨어뜨렸다)
            _dev  = _ev0.get("deviant") or []
            _rare = _ev0.get("rare") or []
            _diff = _ev0.get("differs") or []
            _is_spread = (_ev0.get("spread_tag") == "spread")
            if _dev or _rare:
                _wm = "cluster_mismatch"
            elif _is_spread and len(_diff) >= 2:
                _wm = "neighbor_insufficient"
            else:
                _wm = "accept"
            # ★ sufficiency는 nd만 보는 결정론적 밴드 판정으로, what_missing과
            #   독립된 축이다. 실서빙은 SKIP_ANALYZER_ON_CONFIDENT 때문에
            #   ambiguous일 때만 analyzer를 돌리므로 confident 케이스의 wm_type은
            #   None이다. 오라클도 같은 계약을 유지해야 노트북의
            #   `wm_type.notna()` ambiguous 필터가 그대로 동작한다.
            _suf = ("confident_normal"   if nd_before <  BOUNDARY_LOW else
                    "confident_abnormal" if nd_before >= BOUNDARY_HIGH else "ambiguous")
            if _suf != "ambiguous":
                _wm = None          # confident → 재검색 판단 자체가 불필요
            state["analyzer"] = {
                "wm_type": _wm,
                "sufficiency": _suf,
                "wm_axes": (_ev0.get("deviant") or []) + (_ev0.get("rare") or []),
                "agrees": _ev0.get("agrees") or [],
                "differs": _ev0.get("differs") or [],
                "deviant": _ev0.get("deviant") or [],
                "rare": _ev0.get("rare") or [],
                "persona_match": _ev0.get("persona_match") or [],
                "spread_tag": _ev0.get("spread_tag"),
                "_deterministic": True,
            }
            logger.info("rollout_analyzer_skipped", request_id=request_id,
                        wm_type=_wm, base_spread=_spr,
                        spread_tag=_ev0.get("spread_tag"),
                        n_deviant=len(_dev), n_rare=len(_rare), n_differs=len(_diff),
                        n_agrees=len(_ev0.get("agrees") or []))
        else:
            state.update(await node_analyze(state))    # type: ignore[arg-type]
    except Exception as e:
        logger.error("rollout_base_state_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"rollout base state failed: {e}")

    analyzer_raw = state.get("analyzer") or {}
    base_nview   = _neighbor_view(state.get("neighbors"))
    evidence     = _evidence_from_analyzer(state)

    # candidate pool — agrees ∪ persona_match. 둘 다 비면 categorical 전체로
    # fallback(analyzer 스킵/mock 경로에서도 attribute_knn이 동작해야 한다).
    candidate_pool = list(dict.fromkeys(evidence["agrees"] + evidence["persona_match"]))
    if not candidate_pool and PLANNER_POOL_FALLBACK:
        # ★ [FIX 2] 서빙(_node_plan_local)과 동일한 스위치를 쓴다.
        #   기본값 false — evidence가 비면 오라클도 stop만 남긴다. 예전처럼
        #   여기서만 categorical 전체로 확장하면 오라클은 검색 후보를 갖고
        #   서빙은 stop으로 끝나 학습/서빙 pool이 갈라진다.
        _tx = state.get("transaction") or {}
        candidate_pool = [c for c in _CATEGORICAL_FALLBACK if _tx.get(c) is not None]
        logger.info("rollout_pool_fallback_applied", n=len(candidate_pool))

    # ── step 1 ──
    space1 = _build_step_space(candidate_pool)
    _txk = sorted((state.get("transaction") or {}).keys())
    logger.info("rollout_state_ready", request_id=request_id, test_idx=req.test_idx,
                wm_type=(state.get("analyzer") or {}).get("wm_type"),
                nd_before=round(nd_before, 4),
                pool=candidate_pool, pool_n=len(candidate_pool),
                agrees=evidence["agrees"], differs=evidence["differs"],
                deviant=evidence["deviant"], rare=evidence["rare"],
                persona_match=evidence["persona_match"],
                spread_tag=evidence["spread_tag"],
                tx_keys=_txk,                      # ★ 6개 categorical이 다 왔는가
                n_neighbor_feats=len((base_nview.get("features") or [{}])[0]))
    step1 = await asyncio.gather(*[
        _execute_candidate(c, state, req.y, nd_before, base_nview=base_nview)
        for c in space1
    ])
    step1 = list(step1)
    for i, c in enumerate(step1):
        c["step"] = 1
        c["cand_id"] = f"s1_{i}"
        c["parent_id"] = None
        c["path"] = c["action"] if c["action"] == "stop" else \
            f"{c['action']}({'+'.join(c['features'])})"
        c["n_calls"] = 0.0 if c["action"] == "stop" else 1.0

    candidates: List[Dict] = list(step1)

    # ── step 2 : step1 상위 N개(비 stop)에서만 분기, 같은 tool 반복 금지 ──
    # ★ [1-step 고정] max_steps는 PLANNER_MAX_STEPS를 기본값으로 승계한다.
    #   서빙이 1-step인데 오라클만 2-step을 모으면 학습 모집단과 서빙 경로가
    #   달라진다(노트북도 step==1만 학습 중). 2-step 확장 시 두 값을 함께 올린다.
    if max_steps >= 2:
        parents = sorted([c for c in step1 if c["action"] != "stop" and not c["error"]],
                         key=lambda c: c["delta_margin"], reverse=True)[:step2_topn]
        jobs, metas = [], []
        for p in parents:
            # step1 결과를 base state로 삼는다 — nd/이웃이 갱신된 상태에서 분기
            st2 = dict(state)
            st2["neighbors"] = [
                {"original_index": o, "cluster_id": c, "distance": d, "features": f}
                for o, c, d, f in zip(
                    p["neighbor_oidx"],
                    (p["neighbor_cluster_ids"] or [None] * len(p["neighbor_oidx"])),
                    (p["nd_topk"] or [None] * len(p["neighbor_oidx"])),
                    (p["neighbor_features"] or [{}] * len(p["neighbor_oidx"])),
                )
            ]
            p_nview = _neighbor_view(st2["neighbors"])
            for c2 in _build_step_space(candidate_pool, exclude_tool=p["action"]):
                jobs.append(_execute_candidate(c2, st2, req.y, float(p["nd_after"]),
                                               base_nview=p_nview))
                metas.append((p, c2))
        if jobs:
            step2 = list(await asyncio.gather(*jobs))
            for j, (p, _c2) in enumerate(metas):
                c = step2[j]
                c["step"] = 2
                c["cand_id"] = f"s2_{j}"
                c["parent_id"] = p["cand_id"]
                _leg = c["action"] if c["action"] == "stop" else \
                    f"{c['action']}({'+'.join(c['features'])})"
                c["path"] = f"{p['path']} -> {_leg}"
                c["n_calls"] = p["n_calls"] + (0.0 if c["action"] == "stop" else 1.0)
                # 경로 전체의 순변화 — telescoping (중간 turn reward 소거)
                c["nd_path_before"] = round(nd_before, 6)
                c["delta_margin_path"] = round(
                    _margin(float(c["nd_after"]), req.y) - _margin(nd_before, req.y), 6)
            candidates.extend(step2)

    oracle = (max(candidates, key=lambda c: c.get("delta_margin_path", c["delta_margin"]))
              ["path"] if candidates else None)

    return RolloutResponse(
        request_id=request_id,
        test_idx=req.test_idx,
        y=req.y,
        cluster_id=state.get("cluster_id"),
        wm_type=analyzer_raw.get("wm_type"),
        sufficiency=analyzer_raw.get("sufficiency"),
        wm_axes=analyzer_raw.get("wm_axes"),
        agrees=evidence["agrees"],
        differs=evidence["differs"],
        deviant=evidence["deviant"],
        rare=evidence["rare"],
        persona_match=evidence["persona_match"],
        spread_tag=evidence["spread_tag"],
        nd_before=round(nd_before, 6),
        persona=state.get("persona"),
        neighbors=state.get("neighbors"),
        base_spread=_spread_of(base_nview.get("distances")),
        base_neighbor_cluster_ids=base_nview.get("cluster_ids") or [],
        candidate_pool=candidate_pool,
        candidates=candidates,
        oracle_action=oracle,
        n_steps=max_steps,
        latency_ms=(time.time() - start) * 1000,
    )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")
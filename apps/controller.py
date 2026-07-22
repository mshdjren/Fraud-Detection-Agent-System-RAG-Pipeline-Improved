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
import time
import functools
import json
import operator
import uuid
from typing import Dict, List, Optional, Literal, Tuple, TypedDict, Annotated, Any

from fastapi import FastAPI
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
    "An analyzer already examined a transaction and produced a CLOSED 3-key "
    "verdict: cls (NORMAL/ABNORMAL), sufficiency (confident/ambiguous), and "
    "what_missing (a short free-text hint, only set when sufficiency=ambiguous). "
    "The analyzer's sufficiency already encodes the distance-band judgement — "
    "you must NOT re-derive sufficiency from the numbers yourself; just read it. "
    "DECISION RULE (read in this exact order): "
    "1) If sufficiency=confident, action=accept_verdict — do not second-guess. "
    "2) If sufficiency=ambiguous, read what_missing and SELECT EXACTLY ONE "
    "follow-up tool that matches the kind of gap it describes: sparse-evidence "
    "wording -> expand_within_cluster; cluster/persona-mismatch wording -> "
    "reroute_cross_cluster. "
    "3) what_missing is often just a restatement of the distance band rather "
    "than a new signal — treat it as a WEAK hint, not ground truth. If it is "
    "generic or ambiguous about which gap applies, default to "
    "expand_within_cluster (cheapest follow-up). "
    "4) If expansion_used is already true, action=accept_verdict regardless "
    "(avoid infinite loops). "
    "Do not invent evidence. Keep rationale/hint short. Output closed JSON only."
)


def build_prompt(req: PlanRequest) -> str:
    a = req.analyzer
    tool_catalog = build_followup_tool_catalog()
    persona_summary = req.cluster_persona_summary or "(no profile)"
    return f"""## ANALYZER OUTPUT (3-key schema)
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

## HOW TO DECIDE (sufficiency-first, then what_missing -> tool)
1. sufficiency == "confident"
     -> action=accept_verdict, verdict=cls. STOP here.
2. sufficiency == "ambiguous": read what_missing literally.
   - what_missing mentions sparse evidence / "only top-1 seen" / not enough
     neighbors -> action=expand_within_cluster
   - what_missing mentions cluster/persona mismatch / wrong cluster ->
     action=reroute_cross_cluster
   - what_missing is generic, a distance-band restatement, or unclear ->
     action=expand_within_cluster (cheapest default; do NOT invent a
     mismatch reason that isn't stated)
3. expansion_used == true -> action=accept_verdict regardless of the above
   (loop guard; verdict=cls)

## HINT GENERATION
- agent_hint: imperative single sentence referencing what_missing if present.
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

    # loop 제어
    round:                 int
    budget:                 int
    expansion_used:          bool
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
        updates["top1_distance"] = float(data.get("top1_distance", state.get("top1_distance", 0.5)))
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
    logger.info(
        "node_analyze_calling",
        cluster_id=payload["cluster_id"],
        top_k_count=len(payload["top_k_normal_results"]),
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
    return {
        "analyzer": raw,
        "analyzer_history": [{"round": state.get("round", 0), **normalized.model_dump()}],
        "history": ["analyze"],
        "nd_history": [round(float(_nd_now), 4)],
        "raw_dist_history": [_raw_d],
    }


@instrument_node("plan", required_keys=("request_id", "analyzer"))
async def node_plan(state: AgentState) -> AgentState:
    """controller.py의 /plan 로직(run_plan)을 그대로 graph node로 wrap."""
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
    return {
        "top_k": top_k,
        "cluster_id": target_cluster,
        "top1_distance": data.get("top1_distance"),
        "neighbors": data.get("results", []),
    }


async def _call_tool_mcp(tool_name: str, state: AgentState) -> Dict:
    """
    TOOL_EXECUTION_MODE=mcp — agent_tools_mcp_server.py의 REST shim
    (/tools/{name}/call)을 경유한다. 동일 tool 구현체가 표준 MCP
    streamable-http 경로로도 노출되어 있으므로, 이 경로를 타면 "tool을
    표준 인터페이스 뒤에 둔" 비교군이 된다 (transport overhead 측정 대상).
    """
    payload = _tool_payload(tool_name, state)
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{TOOLS_MCP_URL}/tools/{tool_name}/call", json=payload)
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
    client = MultiServerMCPClient({
        "agent_tools": {"url": f"{TOOLS_MCP_URL}/mcp", "transport": "streamable_http"},
    })
    tools = await client.get_tools()
    tool = next((t for t in tools if t.name == tool_name), None)
    if tool is None:
        raise RuntimeError(f"tool '{tool_name}' not found via MCP tools/list discovery")

    raw = await tool.ainvoke(payload)
    # langchain 도구 호출 결과는 [{"type": "text", "text": "<json string>"}]
    # 형태로 온다(실측 확인됨) — direct/mcp와 동일한 plain dict로 풀어준다.
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
        return json.loads(raw[0]["text"])
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


@instrument_node("expand", required_keys=("cluster_id", "transaction"))
def _apply_patchcore_after_research(updates: Dict, neighbors: List[Dict],
                                    raw_top1: Optional[float]) -> None:
    """재검색(expand/reroute) 결과에 PatchCore reweighting nd를 계산해 updates에 기록.
    round1의 판정 nd는 이 값을 쓴다(min-fusion 폐기 — 재검색했으면 patchcore nd만)."""
    if not PATCHCORE_ON_RESEARCH or raw_top1 is None or not neighbors:
        return
    nd, w, plateau = _patchcore_nd(neighbors, float(raw_top1))
    updates["nd_history"] = [round(nd, 4)]                 # ★ 판정 nd (patchcore)
    updates["raw_dist_history"] = [float(raw_top1)]
    updates["_last_reweight_w"] = w
    updates["_last_plateau"] = plateau
    updates["_nd_source"] = "patchcore_top5"


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
    except Exception as e:
        logger.warning("expand_tool_failed", error=str(e), mode=TOOL_EXECUTION_MODE)
        updates["evidence_top_n"] = EXPAND_TOP_K
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
        if result.get("cluster_id") is not None:
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
    graph.add_node("finalize", node_finalize)

    graph.set_entry_point("pre_router")
    graph.add_conditional_edges("pre_router", route_after_pre_router,
                                {"retrieve": "retrieve", "finalize": "finalize"})
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", "plan")
    graph.add_conditional_edges("plan", route_after_plan, {
        "expand": "expand", "reroute": "reroute", "finalize": "finalize",
    })
    # tool 실행 후 다시 analyze로 (loop) — re-retrieve 후 sufficiency 재평가
    graph.add_edge("expand", "analyze")
    graph.add_edge("reroute", "analyze")
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
    return {"status": "healthy", "service": "planner", "version": "4.0.0-langgraph"}


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
    START([Test Transaction]) --> PR{{"pre_router · fast/deep track"}}
    PR -->|"fast: global dist &lt; τ"| FIN
    PR -->|"deep"| RET["retrieve · Filtered HNSW<br/>거리 top-5 / evidence top-3"]
    RET --> ANA

    subgraph LLMBOX [evaluator · LLM 1회만]
      direction TB
      ANA["analyze · Evaluator LLM<br/><b>what_missing</b> 1-key 생성<br/>cls/suf = 코드(nd) 결정론"]
    end

    ANA --> PLAN{{"plan · Planner<br/>what_missing → tool 매핑"}}
    PLAN -->|"confident"| FIN
    PLAN -->|"wm: sparse-evidence"| EX["expand_within_cluster<br/>같은 cluster top-5 재검색"]
    PLAN -->|"wm: cluster/persona mismatch"| RR["reroute_cross_cluster<br/>rank2 cluster 전환 + persona 교체"]
    EX --> PC["PatchCore reweighting nd<br/>(재검색 top-5 거리 보정)"]
    RR --> PC
    PC -.->|"evaluator 재호출 없음<br/>(EVALUATOR_ONCE)"| FIN
    FIN["finalize<br/>재검색O → patchcore nd / 재검색X → raw nd<br/>nd ≥ CLS_ND_THRESHOLD → ABNORMAL"] --> END([API Response])

    classDef llm fill:#F58518,stroke:#b5610f,color:#fff;
    classDef code fill:#4C78A8,stroke:#2f4b6e,color:#fff;
    classDef term fill:#54A24B,stroke:#356,color:#fff;
    class ANA llm;
    class PR,PLAN,RET,EX,RR,PC,FIN code;
    class START,END term;
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")
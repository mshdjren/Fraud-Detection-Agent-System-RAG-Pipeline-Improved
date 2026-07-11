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
       검증 결과(노트북 Section 9 cell 54) 대부분 distance band의
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
GOOGLE_CLOUD_REGION  = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")

# 서비스 URL (agent loop가 호출하는 외부 마이크로서비스)
RETRIEVER_URL   = os.getenv("RETRIEVER_URL", "http://anomaly-retriever:80")
ANALYZER_URL    = os.getenv("ANALYZER_URL", "http://anomaly-analyzer:80")
TOOLS_MCP_URL   = os.getenv("TOOLS_MCP_URL", "http://agent-tools-mcp:80")
FAST_THRESHOLD  = float(os.getenv("FAST_TRACK_DISTANCE_THRESHOLD", "0.15"))
AGENT_MAX_ROUNDS = int(os.getenv("AGENT_MAX_ROUNDS", "2"))  # expand/reroute는 1회만 허용(루프 가드)

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
    base_top_k_clusters:         Optional[List[Dict]] = None


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

    # retrieve 결과
    top1_distance:       float
    distance_band:       str
    cluster_id:           int
    cluster_persona_summary: Optional[str]
    evidence_top_n:       int               # 현재까지 retrieve한 evidence 개수 (expand가 늘림)

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
            "history": ["retrieve:base_reused"],
        }

    top_n = state.get("evidence_top_n", 1)
    payload = {
        "embedding": state.get("embedding"),
        "transaction": state.get("transaction", {}),
        "top_n": top_n,
    }
    updates: Dict = {"history": ["retrieve"]}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{RETRIEVER_URL}/retrieve", json=payload)
            resp.raise_for_status()
            data = resp.json()
        updates["top1_distance"] = float(data.get("top1_distance", state.get("top1_distance", 0.5)))
        updates["distance_band"] = data.get("distance_band", "unknown")
        updates["cluster_id"] = int(data.get("cluster_id", state.get("cluster_id", -1)))
        updates["cluster_persona_summary"] = data.get("cluster_persona_summary")
    except Exception as e:
        logger.warning("retrieve_failed", error=str(e))
        # retrieve 실패해도 loop는 계속 — 이전 state 값을 변경하지 않음
        # (updates에 안 넣은 필드는 partial dict 반환 시 자동으로 보존됨)
    return updates


async def node_analyze(state: AgentState) -> AgentState:
    """
    Analyzer(Qwen2.5-0.5B sufficiency evaluator) 호출.
    응답이 어떤 schema(3키/6키/그 이후 변형)로 오든 analyzer_schema.
    normalize_analyzer_output()이 표준 표현(NormalizedAnalysis)으로 정규화
    한다 — local_analyzer.py의 출력 schema가 바뀌어도 이 함수는 무수정.
    """
    payload = {
        "transaction":   state.get("transaction", {}),
        "cluster_id":    state.get("cluster_id", -1),
        "top1_distance": state.get("top1_distance", 0.5),
        "evidence_top_n": state.get("evidence_top_n", 1),
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{ANALYZER_URL}/analyze", json=payload)
            resp.raise_for_status()
            raw = resp.json()
    except Exception as e:
        logger.warning("analyze_failed_using_distance_fallback", error=str(e))
        nd = state.get("top1_distance", 0.5)
        suf = "ambiguous" if 0.3 <= nd < 0.55 else "confident"
        raw = {
            "cls": "ABNORMAL" if nd >= 0.5 else "NORMAL",
            "sufficiency": suf,
            "what_missing": "boundary distance; expand evidence" if suf == "ambiguous" else None,
        }
    normalized = AnalyzerOutput(**normalize_analyzer_output(raw).model_dump())
    # ★ analyzer_history(Annotated reducer) 갱신: 이번 round의 analyzer
    # 출력 "1개"만 리스트로 담아 반환 — 이전 round들의 항목은 reducer가
    # 자동으로 이어붙인다. round 번호를 같이 적어둬야 나중에 finalize나
    # 로그에서 "1라운드 sparse → 2라운드 expand 후 confident" 같은 추론
    # 궤적을 시간순으로 재구성할 수 있다.
    return {
        "analyzer": normalized.model_dump(),
        "analyzer_history": [{"round": state.get("round", 0), **normalized.model_dump()}],
        "history": ["analyze"],
    }


async def node_plan(state: AgentState) -> AgentState:
    """controller.py의 /plan 로직(run_plan)을 그대로 graph node로 wrap."""
    analyzer_out = state.get("analyzer", {})
    plan_req = PlanRequest(
        request_id=state.get("request_id"),
        analyzer=AnalyzerOutput(**analyzer_out),
        top1_distance=state.get("top1_distance", 0.5),
        distance_band=state.get("distance_band", "unknown"),
        cluster_id=state.get("cluster_id", -1),
        cluster_persona_summary=state.get("cluster_persona_summary"),
        expansion_used=state.get("expansion_used", False),
    )
    plan_resp = await run_plan(plan_req)
    agent_round_total.labels(action=plan_resp.action).inc()
    # ★ partial dict 반환. action/verdict/agent_hint/planner_backend는
    # reducer가 없는 일반 채널이라 안전하게 덮어쓴다. history는 이번에
    # 추가할 항목 1개만 — analyzer_history는 이 노드가 건드리지 않으므로
    # 아예 키 자체를 반환하지 않는다(이전 버그: state 전체를 반환하면
    # analyzer_history가 이미 채워진 값으로 state 안에 남아있어, reducer가
    # 그 전체를 "이 노드의 추가분"으로 오인해 매번 2배로 불어났다).
    return {
        "action": plan_resp.action,
        "verdict": plan_resp.verdict,
        "agent_hint": plan_resp.agent_hint,
        "planner_backend": plan_resp.backend,
        "history": [f"plan:{plan_resp.action}"],
    }


async def _call_tool_direct(tool_name: str, state: AgentState) -> Dict:
    """
    TOOL_EXECUTION_MODE=direct — agent_tools_mcp_server를 거치지 않고
    Retriever에 바로 HTTP 호출한다 (orchestrator_agentic.py가 기존에
    call_retriever를 재호출하던 패턴과 동일한 의미 — "tool 표준화 레이어
    없이 직접 호출"이 무엇과 비교되는지 그대로 재현하기 위함).
    """
    top_n = state.get("evidence_top_n", 1) + 4
    payload: Dict = {"transaction": state.get("transaction", {}), "top_n": top_n}
    if tool_name == "expand_within_cluster":
        payload["cluster_id"] = state.get("cluster_id", -1)
        payload["mode"] = "within_cluster"
    elif tool_name == "reroute_cross_cluster":
        payload["exclude_cluster_id"] = state.get("cluster_id", -1)
        payload["mode"] = "cross_cluster"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{RETRIEVER_URL}/retrieve", json=payload)
        resp.raise_for_status()
        data = resp.json()
    return {
        "top_n": top_n,
        "cluster_id": data.get("cluster_id"),
        "cluster_persona_summary": data.get("cluster_persona_summary"),
        "neighbors": data.get("neighbors") or data.get("results", []),
    }


async def _call_tool_mcp(tool_name: str, state: AgentState) -> Dict:
    """
    TOOL_EXECUTION_MODE=mcp — agent_tools_mcp_server.py의 REST shim
    (/tools/{name}/call)을 경유한다. 동일 tool 구현체가 표준 MCP
    streamable-http 경로로도 노출되어 있으므로, 이 경로를 타면 "tool을
    표준 인터페이스 뒤에 둔" 비교군이 된다 (transport overhead 측정 대상).
    """
    payload = {
        "cluster_id":     state.get("cluster_id", -1),
        "transaction":     state.get("transaction", {}),
        "top_n":            state.get("evidence_top_n", 1) + 4,  # expand: top-1 -> top-k
        "agent_hint":        state.get("agent_hint"),
    }
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

    payload = {
        "cluster_id":     state.get("cluster_id", -1),
        "transaction":     state.get("transaction", {}),
        "top_n":            state.get("evidence_top_n", 1) + 4,
        "agent_hint":        state.get("agent_hint"),
    }
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


async def node_expand(state: AgentState) -> AgentState:
    updates: Dict = {
        "expansion_used": True,
        "round": state.get("round", 0) + 1,
        "history": [f"expand_within_cluster:{TOOL_EXECUTION_MODE}"],
    }
    try:
        result = await _call_tool("expand_within_cluster", state)
        updates["evidence_top_n"] = result.get("top_n", state.get("evidence_top_n", 1) + 4)
        if result.get("cluster_persona_summary"):
            updates["cluster_persona_summary"] = result["cluster_persona_summary"]
    except Exception as e:
        logger.warning("expand_tool_failed", error=str(e), mode=TOOL_EXECUTION_MODE)
        updates["evidence_top_n"] = state.get("evidence_top_n", 1) + 4
    return updates


async def node_reroute(state: AgentState) -> AgentState:
    updates: Dict = {
        "expansion_used": True,
        "round": state.get("round", 0) + 1,
        "history": [f"reroute_cross_cluster:{TOOL_EXECUTION_MODE}"],
    }
    try:
        result = await _call_tool("reroute_cross_cluster", state)
        if result.get("cluster_id") is not None:
            updates["cluster_id"] = result["cluster_id"]
        if result.get("cluster_persona_summary"):
            updates["cluster_persona_summary"] = result["cluster_persona_summary"]
    except Exception as e:
        logger.warning("reroute_tool_failed", error=str(e), mode=TOOL_EXECUTION_MODE)
    return updates


async def node_finalize(state: AgentState) -> AgentState:
    return {"history": ["finalize"]}


def route_after_pre_router(state: AgentState) -> str:
    return "finalize" if state.get("track") == "fast" else "retrieve"


def route_after_plan(state: AgentState) -> str:
    action = state.get("action", "accept_verdict")
    # 종료 조건: accept_verdict OR round>=budget OR expansion_used(이미 한번 돈 경우)
    if action == "accept_verdict":
        return "finalize"
    if state.get("round", 0) >= state.get("budget", AGENT_MAX_ROUNDS):
        return "finalize"
    if action == "expand_within_cluster":
        return "expand"
    if action == "reroute_cross_cluster":
        return "reroute"
    return "finalize"  # compare_features 등 미구현 action은 안전하게 종료


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
        "evidence_top_n": 1,
        "round": 0,
        "budget": AGENT_MAX_ROUNDS,
        "expansion_used": False,
        "history": [],
        "analyzer_history": [],
    }
    try:
        final_state = await agent_graph.ainvoke(init_state)
        latency_ms = (time.time() - start) * 1000
        analyzer_out = final_state.get("analyzer", {})
        # ★ analyzer_out은 node_analyze에서 AnalyzerOutput(NormalizedAnalysis
        # 상속).model_dump()로 저장된 dict다. .cls/.sufficiency/.what_missing
        # 은 AnalyzerOutput의 @property일 뿐이라 model_dump()에 포함되지
        # 않는다 — 실제 키는 classification/is_sufficient/gap_hint/
        # reasoning_text(analyzer_schema.py의 NormalizedAnalysis 필드명).
        classification = final_state.get("verdict") or analyzer_out.get("classification", "NORMAL")
        trace = final_state.get("history", [])
        planner_used = any(h.startswith("plan:") for h in trace)
        detect_requests_total.labels(status="success", track=final_state.get("track", "deep")).inc()
        detect_latency.observe(latency_ms / 1000)
        return DetectResponse(
            request_id=request_id,
            track=final_state.get("track", "deep"),
            classification=classification,
            confidence=analyzer_out.get("confidence"),
            top_1_distance=final_state.get("top1_distance"),
            primary_cluster_id=final_state.get("cluster_id"),
            reasoning=analyzer_out.get("reasoning_text") or "no_reasoning",
            planner_used=planner_used,
            agent_trace=trace,
            latency_ms=latency_ms,
            tool_execution_mode=TOOL_EXECUTION_MODE,
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

# controller.py에 추가할 엔드포인트
@app.get("/graph/mermaid")
async def graph_mermaid():
    """LangGraph StateGraph를 Mermaid 텍스트로 반환 — 포트폴리오/디버깅용"""
    if agent_graph is None:
        raise HTTPException(status_code=503, detail="graph not built")
    return {"mermaid": agent_graph.get_graph().draw_mermaid()}

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

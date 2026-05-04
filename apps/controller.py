"""
<<<<<<< ours
Controller Service — v1.0 (Layer 1 Single-Shot Planner)
========================================================

목적:
  Router 호출 전, 쿼리당 LLM 1회 호출만으로 파이프라인 plan을 산출.
  Reasoning + loop control + sub-agent 선택을 단일 의사결정자에 통합.

설계 원칙:
  ★ 단일 호출 (예측 가능한 토큰 비용)
  ★ Closed action space (자유 텍스트 plan 금지)
  ★ Fallback plan 필수 (LLM 실패 → 안전한 deep_path 기본값)
  ★ Router 호출 전 정보만 사용
      - global_centroid_distance (단일 dot product, ~ms)
      - 거래 surface feature
      - 시스템 상태 (load, failure rate from caller)

Output Schema (closed):
  {
    "track":            "fast" | "deep",
    "skip_router":      bool,    # fast 시 true
    "sub_agents":       ["distance", "evidence", "persona"],  # deep 시
    "expansion": {
      "enabled":        bool,
      "gap_threshold":  float,   # confidence gap > 이 값이면 router top-N 재검색
      "max_iter":       int,     # 1 권장
      "top_n":          int      # 5 ~ 10
    },
    "tie_break": {
      "enabled":        bool,
      "agreement_threshold": float  # sub-agent 합의 < 이 값이면 LLM tie-break
    },
    "early_exit_confidence": float, # combiner confidence >= 이 값이면 즉시 반환
    "rationale":        str        # 짧은 plan 근거 (관측용)
  }
"""

import os
import time
import json
import uuid
from typing import Dict, List, Optional, Literal
from collections import deque
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import structlog
import asyncio

from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import ChatPromptTemplate

# ===========================
# Configuration
# ===========================

LLM_MODEL            = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
LLM_TEMPERATURE      = float(os.getenv("LLM_TEMPERATURE", "0.0"))   # plan은 결정적이어야 함
LLM_MAX_TOKENS       = int(os.getenv("LLM_MAX_TOKENS", "512"))      # plan은 짧음

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_REGION  = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")

MAX_RETRIES          = int(os.getenv("MAX_RETRIES", "2"))           # plan 단계 retry 보수적
RETRY_BACKOFF        = int(os.getenv("RETRY_BACKOFF", "2"))
LLM_TIMEOUT          = int(os.getenv("LLM_TIMEOUT", "5"))           # plan 호출은 P99 5s 제한

MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))

# Plan 결정 임계값 (LLM에 컨텍스트로 주입)
FAST_TRACK_DIST_THRESHOLD = float(os.getenv("FAST_TRACK_DIST_THRESHOLD", "0.25"))  # 이보다 가까우면 fast 후보
DEEP_TRACK_DIST_THRESHOLD = float(os.getenv("DEEP_TRACK_DIST_THRESHOLD", "0.55"))  # 이보다 멀면 강제 deep

# ===========================
# Rate Limiter (analyzer.py와 동일 패턴)
# ===========================

class RateLimiter:
    def __init__(self, max_calls: int, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        self.lock = Lock()

    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            while self.calls and self.calls[0] < now - self.time_window:
                self.calls.popleft()
            if len(self.calls) >= self.max_calls:
                wait = self.time_window - (now - self.calls[0])
                if wait > 0:
                    time.sleep(wait)
                    now = time.time()
                    while self.calls and self.calls[0] < now - self.time_window:
                        self.calls.popleft()
            self.calls.append(time.time())

rate_limiter = RateLimiter(max_calls=MAX_REQUESTS_PER_MINUTE)

# ===========================
# FastAPI & Logger
# ===========================

app    = FastAPI(title="Anomaly Controller v1.0", version="1.0.0")
logger = structlog.get_logger()

# ===========================
# Prometheus Metrics
# ===========================

plan_requests_total = Counter(
    "plan_requests_total", "Total plan requests", ["status", "track"]
)
plan_latency = Histogram(
    "plan_latency_seconds", "Plan generation latency",
    buckets=[0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
)
plan_fallback_total = Counter(
    "plan_fallback_total", "Fallback plan triggered", ["reason"]
)
sub_agent_selection = Counter(
    "sub_agent_selection_total", "Sub-agent selected by Controller", ["agent_name"]
)
expansion_enabled_total = Counter("expansion_enabled_total", "Plans with expansion enabled")
tie_break_enabled_total = Counter("tie_break_enabled_total", "Plans with tie_break enabled")

# ===========================
# Pydantic Models
# ===========================

class QuerySignal(BaseModel):
    """Router 호출 전 산출 가능한 쿼리 신호."""
    global_centroid_distance: float = Field(..., description="전체 정상 데이터 centroid와의 cosine distance")
    embedding_norm:           float = Field(..., description="임베딩 L2 norm (이상 분포 신호)")
    purchase_value_z:         Optional[float] = Field(None, description="z-score (있으면 surface 신호로 사용)")
    age_z:                    Optional[float] = Field(None, description="z-score")


class SystemState(BaseModel):
    """현재 시스템 상태 — 호출자(orchestrator)가 Prometheus에서 가져와 주입."""
    cpu_load:           float = Field(0.0, ge=0.0, le=1.0, description="0~1, 1=포화")
    recent_failure_rate: float = Field(0.0, ge=0.0, le=1.0, description="최근 1분 실패율")
    recent_p99_ms:      float = Field(0.0, description="최근 1분 P99 latency (ms)")
    cache_hit_rate:     float = Field(0.0, ge=0.0, le=1.0)


class PlanRequest(BaseModel):
    request_id:   Optional[str] = None
    query_signal: QuerySignal
    system_state: SystemState
    # 운영자가 강제로 mode 고정하고 싶을 때 (실험/디버그용)
    force_track:  Optional[Literal["fast", "deep"]] = None


class ExpansionPolicy(BaseModel):
    enabled:       bool  = False
    gap_threshold: float = 0.3
    max_iter:      int   = 1
    top_n:         int   = 5


class TieBreakPolicy(BaseModel):
    enabled:               bool  = False
    agreement_threshold:   float = 0.6


class PlanResponse(BaseModel):
    request_id:    str
    track:         Literal["fast", "deep"]
    skip_router:   bool
    sub_agents:    List[Literal["distance", "evidence", "persona"]]
    expansion:     ExpansionPolicy
    tie_break:     TieBreakPolicy
    early_exit_confidence: float = Field(0.95, ge=0.5, le=1.0)
    rationale:     str
    is_fallback:   bool  = False
    latency_ms:    float

# ===========================
# Vertex AI Client (analyzer.py와 동일 패턴, JSON schema는 plan 전용)
# ===========================

PLAN_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "track":          {"type": "string", "enum": ["fast", "deep"]},
        "skip_router":    {"type": "boolean"},
        "sub_agents": {
            "type": "array",
            "items": {"type": "string", "enum": ["distance", "evidence", "persona"]},
            "maxItems": 3,
        },
        "expansion": {
            "type": "object",
            "properties": {
                "enabled":       {"type": "boolean"},
                "gap_threshold": {"type": "number"},
                "max_iter":      {"type": "integer", "minimum": 0, "maximum": 2},
                "top_n":         {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["enabled", "gap_threshold", "max_iter", "top_n"],
        },
        "tie_break": {
            "type": "object",
            "properties": {
                "enabled":             {"type": "boolean"},
                "agreement_threshold": {"type": "number"},
            },
            "required": ["enabled", "agreement_threshold"],
        },
        "early_exit_confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["track", "skip_router", "sub_agents",
                 "expansion", "tie_break", "early_exit_confidence", "rationale"],
}

try:
    vertexAI = ChatVertexAI(
        model_name=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_TOKENS,
        streaming=False,
        convert_system_message_to_human=True,
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_REGION,
        model_kwargs={
            "response_mime_type": "application/json",
            "response_schema": PLAN_RESPONSE_SCHEMA,
        },
    )
    logger.info("controller_vertex_ai_initialized", model=LLM_MODEL)
except Exception as e:
    logger.error("controller_vertex_ai_init_failed", error=str(e))
    raise

# ===========================
# Prompt
# ===========================

SYSTEM_MSG = (
    "You are a pipeline planner for an anomaly-detection RAG system. "
    "Given a query signal and current system state, output a single JSON plan "
    "that selects: (a) fast vs deep track, (b) which sub-agents to invoke, "
    "(c) whether to enable retrieval expansion, (d) tie-break policy. "
    "Be conservative: prefer 'deep' when uncertain, but escalate cost only when justified by signals."
)


def build_planner_prompt(req: PlanRequest) -> str:
    qs = req.query_signal
    ss = req.system_state

    surface = []
    if qs.purchase_value_z is not None:
        surface.append(f"purchase_value_z={qs.purchase_value_z:.2f}")
    if qs.age_z is not None:
        surface.append(f"age_z={qs.age_z:.2f}")
    surface_str = ", ".join(surface) if surface else "n/a"

    return f"""## QUERY SIGNAL (computed before router)
- global_centroid_distance: {qs.global_centroid_distance:.4f}
- embedding_norm: {qs.embedding_norm:.4f}
- surface_features: {surface_str}

## SYSTEM STATE (current)
- cpu_load: {ss.cpu_load:.2f}   (1.0 = saturated)
- recent_failure_rate: {ss.recent_failure_rate:.2f}
- recent_p99_ms: {ss.recent_p99_ms:.0f}
- cache_hit_rate: {ss.cache_hit_rate:.2f}

## DECISION RULES
- distance < {FAST_TRACK_DIST_THRESHOLD}  AND surface_features 정상 → 'fast' track 후보
  (fast: skip_router=true, sub_agents=[], expansion off, tie_break off)
- distance > {DEEP_TRACK_DIST_THRESHOLD}  → 강제 'deep' track + expansion enabled
- {FAST_TRACK_DIST_THRESHOLD} <= distance <= {DEEP_TRACK_DIST_THRESHOLD} → 'deep' default
  - 강한 surface 이상 신호 있으면 sub_agents에 'evidence' 포함
  - 클러스터 매칭이 애매할 가능성 → 'persona' 포함
  - 항상 'distance' 포함 (값싼 결정론 judge)
- cpu_load > 0.8 OR recent_failure_rate > 0.1 → 비용 절감:
  expansion.max_iter=0, sub_agents 최소화 (distance만)
- recent_p99_ms > 5000 → tie_break.enabled=false (latency 보호)

## OUTPUT
JSON only. No prose outside JSON.
- 'track' = 'fast' or 'deep'
- 'skip_router' = true iff track='fast'
- 'sub_agents' = subset of ['distance','evidence','persona'] (empty if fast)
- 'expansion.gap_threshold' default 0.3
- 'tie_break.agreement_threshold' default 0.6
- 'early_exit_confidence' default 0.95
- 'rationale' = 1 sentence justifying the plan based on signals.
"""

# ===========================
# Fallback Plan
# ===========================

def fallback_plan(reason: str) -> Dict:
    """LLM 실패 / parse 실패 / timeout 시 안전한 기본 plan."""
    plan_fallback_total.labels(reason=reason).inc()
    logger.warning("plan_fallback", reason=reason)
    return {
        "track":       "deep",
        "skip_router": False,
        "sub_agents":  ["distance", "evidence"],
        "expansion":   {"enabled": False, "gap_threshold": 0.3, "max_iter": 0, "top_n": 5},
        "tie_break":   {"enabled": False, "agreement_threshold": 0.6},
        "early_exit_confidence": 0.95,
        "rationale":   f"fallback_plan ({reason})",
    }

# ===========================
# Plan Validation
# ===========================

def validate_and_normalize(plan: Dict, req: PlanRequest) -> Dict:
    """LLM이 schema를 어겨도 안전하게 정규화. 핵심 invariants 강제."""
    # track ↔ skip_router 일관성
    if plan["track"] == "fast":
        plan["skip_router"] = True
        plan["sub_agents"]  = []
        plan["expansion"]["enabled"] = False
        plan["tie_break"]["enabled"] = False
    else:  # deep
        plan["skip_router"] = False
        if not plan.get("sub_agents"):
            plan["sub_agents"] = ["distance"]   # deep인데 비면 최소 distance

    # force_track override
    if req.force_track:
        plan["track"] = req.force_track
        if req.force_track == "fast":
            plan["skip_router"] = True
            plan["sub_agents"]  = []
            plan["expansion"]["enabled"] = False
            plan["tie_break"]["enabled"] = False
        plan["rationale"] = f"force_track={req.force_track}; " + plan.get("rationale", "")

    # 시스템 보호: 부하 높으면 expansion/tie_break 강제 off
    if req.system_state.cpu_load > 0.85 or req.system_state.recent_failure_rate > 0.15:
        plan["expansion"]["enabled"] = False
        plan["expansion"]["max_iter"] = 0
        plan["tie_break"]["enabled"] = False

    # 거리 신호 강제 invariant
    dist = req.query_signal.global_centroid_distance
    if dist > DEEP_TRACK_DIST_THRESHOLD and plan["track"] == "fast":
        # LLM이 잘못 fast 선택한 경우 강제 교정
        plan["track"] = "deep"
        plan["skip_router"] = False
        if not plan["sub_agents"]:
            plan["sub_agents"] = ["distance", "evidence"]
        plan["rationale"] = f"override: dist={dist:.4f} > {DEEP_TRACK_DIST_THRESHOLD}; " + plan.get("rationale", "")

    return plan

# ===========================
# Core: Plan Generation
# ===========================

def _run_planner(req: PlanRequest) -> Dict:
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MSG),
        ("human",  "{human_msg}"),
    ])
    chain = prompt_template | vertexAI

    human_msg = build_planner_prompt(req)
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            rate_limiter.wait_if_needed()
            response = chain.invoke({"human_msg": human_msg})
            text = response.content.strip()
            # JSON fence 제거 (방어적 파싱)
            if text.startswith("```"):
                text = text.lstrip("`").lstrip("json").strip()
                if text.endswith("```"):
                    text = text[:-3].strip()
            return json.loads(text)
        except Exception as e:
            last_error = e
            logger.warning("planner_retry", attempt=attempt + 1, error=str(e))
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))

    raise last_error if last_error else RuntimeError("planner failed")


def _do_plan(req: PlanRequest) -> PlanResponse:
    start = time.time()
    request_id = req.request_id or str(uuid.uuid4())[:8]
    is_fallback = False

    try:
        plan_dict = _run_planner(req)
    except Exception as e:
        logger.error("planner_failed", request_id=request_id, error=str(e))
        plan_dict = fallback_plan(reason="llm_error")
        is_fallback = True

    # 정규화 + invariant 강제
    try:
        plan_dict = validate_and_normalize(plan_dict, req)
    except Exception as e:
        logger.error("plan_validation_failed", request_id=request_id, error=str(e))
        plan_dict = fallback_plan(reason="validation_error")
        is_fallback = True
        plan_dict = validate_and_normalize(plan_dict, req)

    total_ms = (time.time() - start) * 1000
    plan_latency.observe(total_ms / 1000)

    # 메트릭 라벨링
    plan_requests_total.labels(
        status="fallback" if is_fallback else "success",
        track=plan_dict["track"],
    ).inc()
    for agent_name in plan_dict["sub_agents"]:
        sub_agent_selection.labels(agent_name=agent_name).inc()
    if plan_dict["expansion"]["enabled"]:
        expansion_enabled_total.inc()
    if plan_dict["tie_break"]["enabled"]:
        tie_break_enabled_total.inc()

    logger.info(
        "plan_done",
        request_id=request_id,
        track=plan_dict["track"],
        sub_agents=plan_dict["sub_agents"],
        expansion=plan_dict["expansion"]["enabled"],
        tie_break=plan_dict["tie_break"]["enabled"],
        is_fallback=is_fallback,
        latency_ms=total_ms,
    )

    return PlanResponse(
        request_id=request_id,
        track=plan_dict["track"],
        skip_router=plan_dict["skip_router"],
        sub_agents=plan_dict["sub_agents"],
        expansion=ExpansionPolicy(**plan_dict["expansion"]),
        tie_break=TieBreakPolicy(**plan_dict["tie_break"]),
        early_exit_confidence=plan_dict["early_exit_confidence"],
        rationale=plan_dict["rationale"],
        is_fallback=is_fallback,
        latency_ms=total_ms,
    )

# ===========================
# API Endpoints
# ===========================

@app.post("/plan", response_model=PlanResponse)
async def plan(req: PlanRequest):
    try:
        return await asyncio.to_thread(_do_plan, req)
    except Exception as e:
        # 최후의 안전망: 절대 500 안 나도록
        logger.error("plan_endpoint_error", error=str(e))
        plan_dict = fallback_plan(reason="endpoint_error")
        plan_dict = validate_and_normalize(plan_dict, req)
        return PlanResponse(
            request_id=req.request_id or str(uuid.uuid4())[:8],
            track=plan_dict["track"],
            skip_router=plan_dict["skip_router"],
            sub_agents=plan_dict["sub_agents"],
            expansion=ExpansionPolicy(**plan_dict["expansion"]),
            tie_break=TieBreakPolicy(**plan_dict["tie_break"]),
            early_exit_confidence=plan_dict["early_exit_confidence"],
            rationale=plan_dict["rationale"],
            is_fallback=True,
            latency_ms=0.0,
        )


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "controller", "version": "1.0.0"}


@app.get("/ready")
async def ready():
    if vertexAI is None:
        raise HTTPException(status_code=503, detail="Vertex AI not initialized")
    return {
        "status": "ready",
        "model": LLM_MODEL,
        "fast_threshold": FAST_TRACK_DIST_THRESHOLD,
        "deep_threshold": DEEP_TRACK_DIST_THRESHOLD,
    }


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.on_event("startup")
async def startup_event():
    logger.info(
        "controller_started",
        version="1.0.0",
        model=LLM_MODEL,
        fast_threshold=FAST_TRACK_DIST_THRESHOLD,
        deep_threshold=DEEP_TRACK_DIST_THRESHOLD,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")
=======
controller.py
-------------
Single-shot planner controller with lightweight LRU cache.
현재 단계에서는 규칙 기반 fallback을 기본으로 사용하고,
향후 Gemini planner 호출을 같은 인터페이스로 교체할 수 있도록 설계.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class PlanResponse:
    track: str                      # fast | deep
    sub_agents: List[str]           # e.g. ["distance", "evidence", "persona"]
    expansion_enabled: bool
    expansion_budget: int
    tie_break_enabled: bool
    source: str                     # llm | fallback | lru_cache


class PlanLRUCache:
    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self._store: "OrderedDict[Tuple, PlanResponse]" = OrderedDict()

    def get(self, key: Tuple) -> Optional[PlanResponse]:
        if key not in self._store:
            return None
        val = self._store.pop(key)
        self._store[key] = val
        return val

    def put(self, key: Tuple, val: PlanResponse) -> None:
        if key in self._store:
            self._store.pop(key)
        self._store[key] = val
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)


def _bucketize(x: float, step: float = 0.05) -> float:
    return round(x / step) * step


def build_plan_key(*, distance_hint: float, resource_degraded: bool, cache_hot: bool) -> Tuple:
    return (_bucketize(distance_hint), int(resource_degraded), int(cache_hot))


def fallback_plan(*, distance_hint: float, resource_degraded: bool) -> PlanResponse:
    # 시스템 보호 우선
    if resource_degraded:
        return PlanResponse(
            track="fast",
            sub_agents=["distance"],
            expansion_enabled=False,
            expansion_budget=0,
            tie_break_enabled=False,
            source="fallback",
        )

    # 간단한 single-shot plan (향후 LLM 대체 지점)
    if distance_hint >= 0.85:
        return PlanResponse(
            track="fast",
            sub_agents=["distance"],
            expansion_enabled=False,
            expansion_budget=0,
            tie_break_enabled=False,
            source="fallback",
        )
    return PlanResponse(
        track="deep",
        sub_agents=["distance", "evidence", "persona"],
        expansion_enabled=True,
        expansion_budget=1,
        tie_break_enabled=True,
        source="fallback",
    )

>>>>>>> theirs

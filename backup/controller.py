"""
Controller Service — Planner (simplified, initial-stage agent workflow)
=======================================================================
역할: 단일 Planner.
  deep_retrieve → analyzer 의 reasoning + 예비 verdict를 받아,
  "충분한가? 부족하면 어떤 유형인가?"를 진단해 follow-up tool 1개를 선택한다.
    action ∈ FOLLOWUP_ACTIONS = {accept_verdict, expand_within_cluster,
                                 reroute_cross_cluster [, compare_features]}

설계 원칙:
  ★ LLM은 이 한 자리(boundary case의 tool 선택)에서만 개입
  ★ planner는 tool 카탈로그 description을 읽고 reasoning으로 선택
  ★ 결정은 reasoning 텍스트(부족의 '유형')에서, 숫자는 보조 확인용
  ★ Closed action space + closed JSON + 결정론적 fallback

제거된 부가 요소: Layer-1 fast/deep LLM plan, LRU 캐시, MCP/Local/Cascading
                  backend 추상화, CostTracker, rate limiter.
  (fast/deep 분기는 orchestrator의 결정론적 pre-router가 담당)
"""

import os
import time
import json
import uuid
from typing import Dict, List, Optional, Literal, Tuple

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import structlog
import asyncio

# follow-up tool 카탈로그 — agent_tools가 단일 출처. 미배포 시 fallback.
try:
    from agent_tools import build_followup_tool_catalog, FOLLOWUP_ACTIONS
except Exception:
    FOLLOWUP_ACTIONS = (
        "accept_verdict", "expand_within_cluster", "reroute_cross_cluster",
    )
    def build_followup_tool_catalog() -> str:
        return (
            "## AVAILABLE FOLLOW-UP TOOLS (select EXACTLY ONE)\n"
            "- accept_verdict\n- expand_within_cluster\n- reroute_cross_cluster"
        )


# ============================================================
# Configuration
# ============================================================
LLM_MODEL            = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
LLM_TEMPERATURE      = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS       = int(os.getenv("LLM_MAX_TOKENS", "512"))
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_REGION  = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")

app    = FastAPI(title="Anomaly Planner", version="3.0.0-agentic-min")
logger = structlog.get_logger()

# ============================================================
# Metrics
# ============================================================
plan_requests_total = Counter("plan_requests_total", "Planner requests", ["status"])
plan_action_total   = Counter("plan_action_total",   "Planner action chosen", ["action"])
plan_latency        = Histogram("plan_latency_seconds", "Planner latency",
                                buckets=[0.1, 0.3, 0.5, 1.0, 2.0, 5.0])

# ============================================================
# I/O Models
# ============================================================

class AnalyzerOutput(BaseModel):
    classification: Literal["NORMAL", "ABNORMAL"]
    confidence:     int = Field(..., ge=0, le=100)
    reasoning:      str = ""
    key_evidence:   List[str] = Field(default_factory=list)


class PlanRequest(BaseModel):
    request_id:              Optional[str] = None
    analyzer:                AnalyzerOutput
    anomaly_score:           float = Field(0.5, ge=0.0, le=1.0)
    top1_distance:           float = 0.0
    distance_band:           str   = "unknown"
    cluster_id:              int   = -1
    cluster_persona_summary: Optional[str] = None
    expansion_used:          bool  = False


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


# ============================================================
# Vertex AI Client
# ============================================================
try:
    from langchain_google_vertexai import ChatVertexAI
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
            convert_system_message_to_human=True,
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_REGION,
            model_kwargs={
                "response_mime_type": "application/json",
                "response_schema":    PLAN_SCHEMA,
            },
        )
        logger.info("planner_vertex_ai_initialized", model=LLM_MODEL)
    except Exception as e:
        logger.error("planner_vertex_ai_init_failed", error=str(e))
        vertexAI = None

# ============================================================
# Prompt
# ============================================================

SYSTEM_MSG = (
    "You are the Planner for a fraud-detection RAG agent. "
    "An analyzer already examined a transaction using the matched normal cluster, "
    "its nearest normal neighbors, the neighbor features, and the cluster persona, then "
    "produced reasoning + a preliminary classification. "
    "Your ONLY job: read the analyzer reasoning, DIAGNOSE whether the verdict is trustworthy, "
    "and if not, identify *what kind* of evidence is missing, then SELECT EXACTLY ONE follow-up tool. "
    "Decide from the TEXTUAL cues in the reasoning, not from the numbers alone "
    "(the numbers are a secondary sanity check). "
    "Match the type of uncertainty to the tool: sparse evidence -> expand, "
    "cluster/persona mismatch -> reroute, missing reason-why -> compare. "
    "Do not invent evidence. Keep rationale/hint short. Output closed JSON only."
)


def build_prompt(req: PlanRequest) -> str:
    a = req.analyzer
    tool_catalog = build_followup_tool_catalog()
    persona_summary = req.cluster_persona_summary or "(no profile)"
    return f"""## ANALYZER OUTPUT
- classification: {a.classification}
- confidence:     {a.confidence}
- reasoning:      {a.reasoning.strip()}
- key_evidence:   {a.key_evidence}

## CONTEXT SIGNALS (secondary; do NOT threshold on these alone)
- anomaly_score:  {req.anomaly_score:.4f}
- top1_distance:  {req.top1_distance:.4f}
- distance_band:  {req.distance_band}
- cluster_id:     {req.cluster_id}
- cluster_persona: {persona_summary}
- expansion_used: {req.expansion_used}

{tool_catalog}

## HOW TO DECIDE — diagnose the CAUSE of uncertainty, then map to ONE tool
- Reasoning is confident and self-consistent
    -> action=accept_verdict (set verdict to the analyzer classification)
- Reasoning complains evidence is too sparse / cannot pin the normal range
    -> action=expand_within_cluster
- Reasoning says the sample does not fit this cluster / persona mismatch
    -> action=reroute_cross_cluster
- Enough neighbors, but reasoning lacks a concrete REASON why it is abnormal
    -> action=compare_features
- expansion_used is already true -> prefer accept_verdict (avoid loops)

## HINT GENERATION
- agent_hint: imperative single sentence referencing SPECIFIC features named in the reasoning.
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
        verdict=req.analyzer.classification,
        expansion=ExpansionPolicy(),
        agent_hint=None,
        rationale=f"fallback ({reason}): accept analyzer verdict",
        backend="fallback",
        is_fallback=True,
    )


# ============================================================
# Core
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
        verdict = req.analyzer.classification

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


@app.post("/plan", response_model=PlanResponse)
async def plan(req: PlanRequest):
    start = time.time()
    try:
        out, backend = await _call_llm(req)
    except Exception as e:
        logger.error("planner_llm_failed", error=str(e))
        return fallback_plan(req, reason="llm_error")
    try:
        latency_ms = (time.time() - start) * 1000
        result = _normalize(out, req, backend, latency_ms)
        plan_requests_total.labels(status="success").inc()
        plan_latency.observe(latency_ms / 1000)
        logger.info("plan_done", request_id=result.request_id, action=result.action,
                    backend=backend, latency_ms=latency_ms)
        return result
    except Exception as e:
        logger.error("planner_validation_failed", error=str(e))
        return fallback_plan(req, reason="validation_error")


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "planner", "version": "3.0.0-agentic-min"}


@app.get("/ready")
async def ready():
    return {"status": "ready", "model": LLM_MODEL,
            "vertex_ready": vertexAI is not None,
            "actions": list(FOLLOWUP_ACTIONS)}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.on_event("startup")
async def startup_event():
    logger.info("planner_started", model=LLM_MODEL, actions=list(FOLLOWUP_ACTIONS))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")

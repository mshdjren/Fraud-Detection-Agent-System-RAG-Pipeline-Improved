"""
orchestrator_agentic.py — Agent workflow system (simplified, initial stage)
===========================================================================
다이어그램 그대로의 최소 agent workflow:

[deterministic pre-router] → fast track(numerical, 종료) │ deep track
                                                          ↓
                                             [deep_retrieve] (base)
                                                          ↓  (cluster+neighbor+features+persona)
                                             [analyzer] → reasoning + 예비 verdict
                                                          ↓
                                             [planner] ── 충분 ──→ [finalize]
                                                 │
                                                 └─ 부족 → 유형 진단 → {expand | reroute [| compare]}
                                                              ↓ (augmented evidence로 analyzer 재실행)
                                                         └──── loop (max N회) ────┘

단일 analyzer(view 분할 없음), planner는 boundary case에서만 개입.
제거된 부가 요소: view dispatch, multi-judge Stouffer, LRU/prefix 캐시,
                  CostTracker, MCP wrapper 분기, PyOD/tree_features,
                  Layer-1 LLM plan(→ 결정론적 pre-router로 대체).
"""

import os
import time
import math
import asyncio
import uuid
from typing import Dict, List, Optional, Literal

import httpx
import numpy as np
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import structlog
from pybreaker import CircuitBreaker

from agent_tools import (
    decide_track,
    compute_feature_comparison,
    FOLLOWUP_ACTIONS,
)

# ============================================================
# Configuration
# ============================================================
ROUTER_URL     = os.getenv("ROUTER_URL",     "http://anomaly-router:80")
RETRIEVER_URL  = os.getenv("RETRIEVER_URL",  "http://anomaly-retriever:80")
ANALYZER_URL   = os.getenv("ANALYZER_URL",   "http://local-analyzer:80")
CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://anomaly-controller:80")

TOOL_EXECUTION_MODE = os.getenv("TOOL_EXECUTION_MODE", "direct")

ROUTER_TIMEOUT     = int(os.getenv("ROUTER_TIMEOUT",     "5"))
RETRIEVER_TIMEOUT  = int(os.getenv("RETRIEVER_TIMEOUT",  "10"))
ANALYZER_TIMEOUT   = int(os.getenv("ANALYZER_TIMEOUT",   "300"))
PLANNER_TIMEOUT    = int(os.getenv("PLANNER_TIMEOUT",    "8"))

# pre-router 임계값 (fast/deep)
FAST_TRACK_DIST_THRESHOLD = float(os.getenv("FAST_TRACK_DIST_THRESHOLD", "0.25"))

# uncertainty zone (boundary) → planner 개입 구간
BOUNDARY_LOW  = float(os.getenv("BOUNDARY_LOW",  "0.35"))
BOUNDARY_HIGH = float(os.getenv("BOUNDARY_HIGH", "0.65"))

# planner 루프 최대 반복 (circuit breaker 역할)
MAX_EXPANSION_ITER = int(os.getenv("MAX_EXPANSION_ITER", "1"))

# 점수 결합 가중 (analyzer vs distance) — 단순 weighted average
W_ANALYZER = float(os.getenv("W_ANALYZER", "0.6"))

GLOBAL_CENTROID_PATH = os.getenv("GLOBAL_CENTROID_PATH", "/data/global_centroid.npy")

# retry / circuit breaker
MAX_RETRIES          = int(os.getenv("MAX_RETRIES",          "5"))
RETRY_BACKOFF_FACTOR = int(os.getenv("RETRY_BACKOFF_FACTOR", "4"))
RETRY_INITIAL_WAIT   = int(os.getenv("RETRY_INITIAL_WAIT",   "2"))
CB_FAILURE_THRESHOLD = int(os.getenv("CB_FAILURE_THRESHOLD", "10"))
CB_RECOVERY_TIMEOUT  = int(os.getenv("CB_RECOVERY_TIMEOUT",  "60"))

# ============================================================
# FastAPI & Logger & Metrics
# ============================================================
app = FastAPI(title="Anomaly Orchestrator Agentic", version="7.0.0-agentic-min")
logger = structlog.get_logger()

track_total          = Counter("track_total", "Pre-router track", ["track"])
boundary_hit_total   = Counter("boundary_hit_total", "Boundary (uncertain) cases")
planner_calls_total  = Counter("planner_calls_total", "Planner calls", ["status"])
action_total         = Counter("action_total", "Follow-up action executed", ["action"])
feedback_iteration   = Counter("feedback_iteration_total", "Planner loop iterations")
service_latency      = Histogram("service_latency_seconds", "Service latency", ["service"],
                                 buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0])
retry_attempts       = Counter("retry_attempts_total", "Retry attempts", ["service", "attempt"])

global_centroid: Optional[np.ndarray] = None

# ============================================================
# Models
# ============================================================
validation_errors = Counter("validation_errors_total", "Validation errors", ["field"])


class TestCase(BaseModel):
    """파이프라인 입력. embedding: AnoLLM 576-dim (Router 내부에서 PCA/전처리)."""
    purchase_value: float
    age:            float
    sex:            Literal["M", "F"]
    source:         Literal["Direct", "Ads", "SEO"]
    browser:        Literal["Chrome", "IE", "Safari", "FireFox", "Opera"]
    embedding:      List[float]
    experiment_case:   Optional[str] = None
    percolate_version: Optional[str] = None

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v):
        if len(v) != 576:
            validation_errors.labels(field="embedding").inc()
            raise ValueError(f"embedding must be 576-dim, got {len(v)}")
        return v

    class Config:
        extra = "ignore"


class ClusterCandidate(BaseModel):
    cluster_id: int
    rank: int
    score: float


class PipelineResponse(BaseModel):
    classification: str
    confidence:     int
    reasoning:      Optional[str]
    key_evidence:   List[str]
    final_verdict:  Optional[str]
    primary_cluster_id: int
    top_5_clusters:     List[ClusterCandidate]
    vec_index:          str
    match_type:         str
    match_score:        float
    top_1_distance:     float
    top_k:              int
    track:              str           # "fast" | "deep"
    planner_used:       bool
    agent_trace:        List[str]     # 실행된 tool 시퀀스
    latency_ms:         Dict[str, float]
    experiment_case:    str
    percolate_version:  str


# ============================================================
# Helpers
# ============================================================

def distance_to_score(distance: float, threshold: float = 0.5) -> float:
    x = (float(distance) - threshold) * 5
    return round(1 / (1 + math.exp(-x)), 4)


def compute_global_distance(embedding: List[float]) -> float:
    if global_centroid is None:
        return min(2.0, math.sqrt(sum(v * v for v in embedding[:64])) / 8.0)
    v = np.array(embedding, dtype=np.float32)
    denom = (np.linalg.norm(v) * np.linalg.norm(global_centroid) + 1e-8)
    cos = float(np.dot(v, global_centroid) / denom)
    return float(1.0 - cos)


def coarse_band(d: float) -> str:
    d = float(d)
    if d < 0.30:
        return "close"
    if d < 0.60:
        return "boundary"
    return "far"


def analyzer_to_score(classification: str, confidence: int) -> float:
    """판정 + confidence → anomaly score (0~1, 1=ABNORMAL)."""
    p = max(0, min(100, int(confidence))) / 100.0
    return p if classification == "ABNORMAL" else (1.0 - p)


def combine_score(analyzer_score: float, distance_score: float, *, w: float = W_ANALYZER) -> float:
    """단순 weighted average (Stouffer/disagreement penalty 제거 — 단일 analyzer)."""
    return max(0.0, min(1.0, w * float(analyzer_score) + (1.0 - w) * float(distance_score)))


# ============================================================
# HTTP with retry
# ============================================================
async def call_with_retry(client, method, url, timeout, service_name="unknown", **kwargs):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.request(method=method, url=url, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except httpx.HTTPError as e:
            last_error = e
            if hasattr(e, "response") and e.response is not None and e.response.status_code < 500:
                raise
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_INITIAL_WAIT * (RETRY_BACKOFF_FACTOR ** attempt)
                retry_attempts.labels(service=service_name, attempt=str(attempt + 1)).inc()
                logger.warning("retry", service=service_name, attempt=attempt + 1, wait=wait)
                await asyncio.sleep(wait)
            else:
                raise last_error
    raise last_error


# ============================================================
# Circuit breakers + service calls
# ============================================================
router_breaker    = CircuitBreaker(fail_max=CB_FAILURE_THRESHOLD, reset_timeout=CB_RECOVERY_TIMEOUT, name="router")
retriever_breaker = CircuitBreaker(fail_max=CB_FAILURE_THRESHOLD, reset_timeout=CB_RECOVERY_TIMEOUT, name="retriever")
analyzer_breaker  = CircuitBreaker(fail_max=CB_FAILURE_THRESHOLD, reset_timeout=CB_RECOVERY_TIMEOUT, name="analyzer")


@router_breaker
async def call_router(client, embedding, experiment_case, percolate_version) -> Dict:
    start = time.time()
    resp = await call_with_retry(
        client, "POST", f"{ROUTER_URL}/route", timeout=ROUTER_TIMEOUT, service_name="router",
        json={"embedding": embedding},
        headers={"Content-Type": "application/json",
                 "X-Router-Case": experiment_case, "X-Router-Version": percolate_version},
    )
    result = resp.json()
    service_latency.labels(service="router").observe(time.time() - start)
    logger.info("router_success", primary_cluster_id=result.get("primary_cluster_id"),
                vec_index=result.get("vec_index"), match_type=result.get("match_type"))
    return result


@retriever_breaker
async def call_retriever(client, embedding, cluster_ids, vec_index, top_k=5) -> Dict:
    start = time.time()
    resp = await call_with_retry(
        client, "POST", f"{RETRIEVER_URL}/retrieve", timeout=RETRIEVER_TIMEOUT, service_name="retriever",
        json={"embedding": embedding, "cluster_ids": cluster_ids, "vec_index": vec_index, "top_k": top_k},
        headers={"Content-Type": "application/json"},
    )
    result = resp.json()
    service_latency.labels(service="retriever").observe(time.time() - start)
    logger.info("retriever_success", cluster_ids=cluster_ids,
                n_results=len(result.get("results", [])), top1_distance=result.get("top1_distance"))
    return result


@analyzer_breaker
async def call_analyzer(client, test_transaction, cluster_id, top_k_normal_results,
                        top1_distance, experiment_case,
                        persona: Optional[Dict] = None, agent_hint: Optional[str] = None) -> Dict:
    """
    단일 analyzer 호출 — cluster + neighbors + features + persona 전부를 받아
    reasoning + 예비 verdict 산출. agent_hint는 재실행 시 planner가 준 힌트(미지원 시 무시).
    """
    start = time.time()
    body: Dict = {
        "test_transaction":     test_transaction,
        "cluster_id":           cluster_id,
        "top_k_normal_results": top_k_normal_results,
        "top1_distance":        top1_distance,
    }
    if persona is not None:
        body["persona"] = persona
    if agent_hint:
        body["agent_hint"] = agent_hint

    resp = await call_with_retry(
        client, "POST", f"{ANALYZER_URL}/analyze", timeout=ANALYZER_TIMEOUT, service_name="analyzer",
        json=body, headers={"Content-Type": "application/json", "X-Experiment-Case": experiment_case},
    )
    result = resp.json()
    service_latency.labels(service="analyzer").observe(time.time() - start)
    logger.info("analyzer_success", classification=result.get("classification"),
                confidence=result.get("confidence"))
    return result


async def call_planner(client, *, request_id, analyzer, anomaly_score, top1_distance,
                       distance_band, cluster_id, cluster_persona_summary, expansion_used) -> Dict:
    """
    planner(/plan) 호출. CONTROLLER_URL이 비었거나 실패하면 결정론적 fallback(accept).
    """
    fallback = {
        "action": "accept_verdict",
        "verdict": analyzer.get("classification", "NORMAL"),
        "expansion": {"top_n": 5},
        "agent_hint": None,
        "rationale": "fallback: accept analyzer verdict",
        "backend": "fallback",
    }
    if not CONTROLLER_URL:
        planner_calls_total.labels(status="no_url").inc()
        return fallback

    payload = {
        "request_id": request_id,
        "analyzer": {
            "classification": analyzer.get("classification", "NORMAL"),
            "confidence":     int(analyzer.get("confidence", 50)),
            "reasoning":      analyzer.get("reasoning", "") or "",
            "key_evidence":   analyzer.get("key_evidence", []),
        },
        "anomaly_score":  anomaly_score,
        "top1_distance":  top1_distance,
        "distance_band":  distance_band,
        "cluster_id":     cluster_id,
        "cluster_persona_summary": cluster_persona_summary,
        "expansion_used": expansion_used,
    }
    try:
        resp = await client.post(f"{CONTROLLER_URL}/plan", json=payload, timeout=PLANNER_TIMEOUT)
        resp.raise_for_status()
        planner_calls_total.labels(status="success").inc()
        return resp.json()
    except Exception as e:
        logger.error("planner_failed", error=str(e))
        planner_calls_total.labels(status="fallback").inc()
        return fallback


# ============================================================
# Main pipeline
# ============================================================
@app.post("/detect", response_model=PipelineResponse)
async def detect(tc: TestCase):
    start = time.time()
    request_id = uuid.uuid4().hex[:8]
    lat: Dict[str, float] = {}
    tx = {"purchase_value": tc.purchase_value, "age": tc.age, "sex": tc.sex,
          "source": tc.source, "browser": tc.browser}
    experiment_case   = tc.experiment_case   or "pca_64_k100"
    percolate_version = tc.percolate_version or "v14"

    async with httpx.AsyncClient() as client:
        # ── 1) Deterministic pre-router ──
        global_dist = compute_global_distance(tc.embedding)
        track = decide_track(global_dist, fast_threshold=FAST_TRACK_DIST_THRESHOLD)
        track_total.labels(track=track).inc()

        # ── 2) Fast track (numerical, 종료) ──
        if track == "fast":
            score = distance_to_score(global_dist)
            return PipelineResponse(
                classification="ABNORMAL" if score >= 0.5 else "NORMAL",
                confidence=int(score * 100),
                reasoning="fast_track (numerical)",
                key_evidence=[f"global_distance={global_dist:.4f}"],
                final_verdict=None,
                primary_cluster_id=-1, top_5_clusters=[], vec_index="",
                match_type="fast", match_score=0.0,
                top_1_distance=global_dist, top_k=0,
                track="fast", planner_used=False, agent_trace=["pre_router:fast"],
                latency_ms={"total": (time.time() - start) * 1000},
                experiment_case=experiment_case, percolate_version=percolate_version,
            )

        # ── 3) deep_retrieve (base): router + retriever ──
        r0 = await call_router(client, tc.embedding, experiment_case, percolate_version)
        primary_cluster_id = r0["primary_cluster_id"]
        clusters = [c["cluster_id"] for c in r0.get("top_k_clusters", [])]
        rr = await call_retriever(client, tc.embedding, clusters, r0["vec_index"])
        top1 = rr.get("top1_distance", 999.0)
        results = rr.get("results", [])
        persona = r0.get("persona")
        persona_summary = persona.get("description") if isinstance(persona, dict) else None
        agent_trace: List[str] = ["pre_router:deep", "deep_retrieve"]

        # ── 4) analyzer (single, 전체 정보) ──
        analyzer = await call_analyzer(
            client, tx, primary_cluster_id, results, top1, experiment_case, persona=persona
        )
        agent_trace.append("analyzer")

        # ━━━ [SMOKE DEBUG] planner 강제 개입용 override ━━━
        # 환경변수 FORCE_BOUNDARY=true 시 analyzer 결과를 boundary zone으로 덮어씀
        # if os.getenv("FORCE_BOUNDARY", "false").lower() == "true":
        analyzer["classification"] = "ABNORMAL"
        analyzer["confidence"]     = 40
        analyzer["reasoning"]      = (
            "[FORCED] Evidence is too sparse. "
            "Cannot pin the normal range with current neighbors. "
            "Need more neighbors from the same cluster."
        )
        analyzer["key_evidence"]   = ["forced_boundary=true", "sparse_evidence_hint"]
        logger.info("forced_boundary_override")

        score = combine_score(
            analyzer_to_score(analyzer.get("classification", "NORMAL"), int(analyzer.get("confidence", 50))),
            distance_to_score(top1),
        )

        # ── 5) Planner loop (boundary일 때만, max N회) ──
        planner_used = False
        expansion_used = False
        for it in range(MAX_EXPANSION_ITER + 1):
            is_boundary = BOUNDARY_LOW <= score <= BOUNDARY_HIGH
            if not is_boundary:
                break                       # 충분 → finalize
            if it >= MAX_EXPANSION_ITER:
                break                       # circuit breaker: 더 이상 확장 안 함
            boundary_hit_total.inc()

            planner_used = True
            plan = await call_planner(
                client, request_id=request_id, analyzer=analyzer,
                anomaly_score=score, top1_distance=top1, distance_band=coarse_band(top1),
                cluster_id=primary_cluster_id, cluster_persona_summary=persona_summary,
                expansion_used=expansion_used,
            )
            action = plan.get("action", "accept_verdict")
            if action not in FOLLOWUP_ACTIONS:
                action = "accept_verdict"
            action_total.labels(action=action).inc()
            agent_trace.append(action)

            if action == "accept_verdict":
                break

            feedback_iteration.inc()
            top_n = max(1, min(20, int((plan.get("expansion") or {}).get("top_n", 5))))
            hint = plan.get("agent_hint")

            # ── 부족 유형별 tool 실행 (재질의) ──
            if action == "expand_within_cluster":
                rr2 = await call_retriever(client, tc.embedding, [primary_cluster_id], r0["vec_index"], top_k=top_n)
                top1 = rr2.get("top1_distance", top1)
                results = rr2.get("results", results)
            elif action == "reroute_cross_cluster":
                others = [cid for cid in clusters if cid != primary_cluster_id] or clusters
                rr2 = await call_retriever(client, tc.embedding, others, r0["vec_index"], top_k=top_n)
                top1 = rr2.get("top1_distance", top1)
                results = rr2.get("results", results)
            elif action == "compare_features":
                nn = results[0].get("original_features") if results and isinstance(results[0], dict) else None
                cmp = compute_feature_comparison(tx, nn)
                hint = (hint + " " + cmp["hint"]).strip() if hint else cmp["hint"]

            # ── augmented evidence로 analyzer 재실행 ──
            analyzer = await call_analyzer(
                client, tx, primary_cluster_id, results, top1, experiment_case,
                persona=persona, agent_hint=hint,
            )
            agent_trace.append("analyzer")
            score = combine_score(
                analyzer_to_score(analyzer.get("classification", "NORMAL"), int(analyzer.get("confidence", 50))),
                distance_to_score(top1),
            )
            expansion_used = True

        # ── 6) Finalize ──
        agent_trace.append("finalize")
        final_classification = "ABNORMAL" if score >= 0.5 else "NORMAL"
        lat["total"] = (time.time() - start) * 1000
        return PipelineResponse(
            classification=final_classification,
            confidence=int(round(score * 100)),
            reasoning=analyzer.get("reasoning") or "no_reasoning",
            key_evidence=list(analyzer.get("key_evidence", [])) + [
                f"track=deep", f"score={score:.3f}", f"top1={top1:.3f}",
                f"planner_used={planner_used}",
            ],
            final_verdict=analyzer.get("final_verdict"),
            primary_cluster_id=primary_cluster_id,
            top_5_clusters=[ClusterCandidate(**c) for c in r0.get("top_k_clusters", [])],
            vec_index=r0["vec_index"],
            match_type=r0.get("match_type", "strict"),
            match_score=float(r0.get("match_score", 1.0)),
            top_1_distance=top1, top_k=len(results),
            track="deep", planner_used=planner_used, agent_trace=agent_trace,
            latency_ms=lat,
            experiment_case=experiment_case, percolate_version=percolate_version,
        )


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "orchestrator-agentic",
            "actions": list(FOLLOWUP_ACTIONS), "max_iter": MAX_EXPANSION_ITER}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.on_event("startup")
async def startup_event():
    global global_centroid
    if os.path.exists(GLOBAL_CENTROID_PATH):
        try:
            global_centroid = np.load(GLOBAL_CENTROID_PATH).astype(np.float32)
        except Exception:
            global_centroid = None
    logger.info("orchestrator_started", actions=list(FOLLOWUP_ACTIONS),
                fast_threshold=FAST_TRACK_DIST_THRESHOLD,
                boundary=[BOUNDARY_LOW, BOUNDARY_HIGH], max_iter=MAX_EXPANSION_ITER)

@app.get("/ready")
async def ready():
    try:
        async with httpx.AsyncClient() as client:
            checks = [
                ("router",    f"{ROUTER_URL}/health"),
                ("retriever", f"{RETRIEVER_URL}/health"),
                ("analyzer",  f"{ANALYZER_URL}/health"),
            ]
            if CONTROLLER_URL:
                checks.append(("planner", f"{CONTROLLER_URL}/ready"))

            results = {}
            for svc, url in checks:
                try:
                    r = await client.get(url, timeout=2)
                    results[svc] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
                except Exception as e:
                    results[svc] = f"error: {e}"

            if not CONTROLLER_URL:
                results["planner"] = "skipped (deterministic fallback)"

        all_ok = all(v == "ok" or v.startswith("skipped") for v in results.values())
        if not all_ok:
            raise HTTPException(status_code=503, detail=results)

        return {
            "status":  "ready",
            "version": "7.0.0-agentic-min",
            "services": results,
            "configuration": {
                "fast_track_dist_threshold": FAST_TRACK_DIST_THRESHOLD,
                "boundary_zone":             [BOUNDARY_LOW, BOUNDARY_HIGH],
                "max_expansion_iter":        MAX_EXPANSION_ITER,
                "w_analyzer":                W_ANALYZER,
                "tool_execution_mode":       TOOL_EXECUTION_MODE,
                "controller_url":            CONTROLLER_URL or "(not set)",
                "followup_actions":          list(FOLLOWUP_ACTIONS),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

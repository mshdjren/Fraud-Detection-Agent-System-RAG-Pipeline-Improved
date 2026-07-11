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
    compute_feature_comparison,  # (참고용 — agent loop 본체는 controller.py로 이전, 여기선 미사용)
    FOLLOWUP_ACTIONS,
)

# ============================================================
# Configuration
# ============================================================
ROUTER_URL     = os.getenv("ROUTER_URL",     "http://anomaly-router:80")
RETRIEVER_URL  = os.getenv("RETRIEVER_URL",  "http://anomaly-retriever:80")
ANALYZER_URL   = os.getenv("ANALYZER_URL",   "http://local-analyzer:80")   # (참고용 — controller가 직접 호출, orchestrator는 미사용)
CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://anomaly-controller:80")

TOOL_EXECUTION_MODE = os.getenv("TOOL_EXECUTION_MODE", "direct")

ROUTER_TIMEOUT     = int(os.getenv("ROUTER_TIMEOUT",     "5"))
RETRIEVER_TIMEOUT  = int(os.getenv("RETRIEVER_TIMEOUT",  "10"))
PLANNER_TIMEOUT    = int(os.getenv("PLANNER_TIMEOUT",    "8"))
# ANALYZER_TIMEOUT 제거됨 — analyzer 호출은 controller.py가 직접 수행하므로
# orchestrator에는 더 이상 해당 타임아웃이 필요 없다.

# pre-router 임계값 (fast/deep) — 유일하게 orchestrator가 직접 판정하는 분기점.
FAST_TRACK_DIST_THRESHOLD = float(os.getenv("FAST_TRACK_DIST_THRESHOLD", "0.25"))
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
app = FastAPI(title="Anomaly Orchestrator Agentic", version="8.0.0-agentic-thin")
logger = structlog.get_logger()

track_total          = Counter("track_total", "Pre-router track", ["track"])
planner_calls_total  = Counter("planner_calls_total", "Controller agent-loop delegation calls", ["status"])
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
    # ★ [Observability ③] controller round_log passthrough — harness Layer3 입력
    agent_rounds:       List[Dict] = []
    termination:        Optional[str] = None
    experiment_case:    str
    percolate_version:  str


# ============================================================
# Helpers
# ============================================================

def distance_to_score(distance: float, threshold: float = 0.5) -> float:
    x = (float(distance) - threshold) * 5
    return round(1 / (1 + math.exp(-x)), 4)

DIST_P50 = float(os.getenv("DIST_P50", "1.448"))
DIST_IQR = float(os.getenv("DIST_IQR", "0.387"))

def _normalize_dist(d: float) -> float:
    """local_analyzer.py의 prompts.normalize_dist와 동일 수식.
    retriever가 반환하는 top1_distance는 raw L2(~1.2~1.7)이므로, 0~1 스케일
    임계값(coarse_band, distance_to_score)과 비교하기 전에 정규화가 필요하다
    — 이전엔 정규화 없이 비교해서 coarse_band가 항상 'far'로 나오던 버그가 있었다."""
    if DIST_IQR == 0:
        return 0.5
    return 1 / (1 + math.exp(-(d - DIST_P50) / DIST_IQR))


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
# analyzer_breaker / call_analyzer 제거됨 — analyzer 호출은 이제
# controller.py(/detect, LangGraph analyze node)가 직접 수행한다.
# orchestrator는 router+retriever "base call"까지만 담당한다.


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
async def call_retriever(client, embedding, cluster_ids, vec_index, top_k=3) -> Dict:  # ★ 기본 top-k=3
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


async def call_controller_agent_loop(
    client, *, request_id: str, transaction: Dict, embedding: List[float],
    global_distance: float, cluster_id: int, top1_distance: float,
    distance_band: str, cluster_persona_summary: Optional[str],
    base_neighbors: Optional[List[Dict]] = None,
    base_vec_index: Optional[str] = None,
    base_top_k_clusters: Optional[List[Dict]] = None,  # ★ router top-5 (persona 포함) — reroute 소스
    base_persona: Optional[Dict] = None,               # ★ primary persona dict — analyzer 입력 복원
) -> Dict:
    """
    controller.py 의 /detect (LangGraph StateGraph: analyze→plan→tool→loop)
    로 agent loop 전체를 위임한다. orchestrator는 router+retriever "base
    call"을 이미 끝낸 결과(cluster_id/top1_distance/...)를 base_* 필드로
    동봉해, controller가 동일 retrieve를 중복 수행하지 않도록 한다.

    CONTROLLER_URL이 비었거나 실패하면, agent loop 없이 base retrieve
    결과만으로 결정론적 verdict를 만드는 fallback으로 내려간다(분석/계획
    없이 distance band로만 판정 — "agent loop가 죽어도 서비스는 응답한다"
    원칙).
    """
    fallback = {
        "classification": "ABNORMAL" if _normalize_dist(top1_distance) >= 0.5 else "NORMAL",
        "confidence": int(round(distance_to_score(_normalize_dist(top1_distance), threshold=0.5) * 100)),
        "reasoning": "fallback: controller unavailable, distance-only verdict",
        "planner_used": False,
        "agent_trace": ["controller_unavailable_fallback"],
        "tool_execution_mode": TOOL_EXECUTION_MODE,
    }
    if not CONTROLLER_URL:
        planner_calls_total.labels(status="no_url").inc()
        logger.warning("controller_url_not_configured", request_id=request_id)
        return fallback

    payload = {
        "request_id": request_id,
        "transaction": transaction,
        "embedding": embedding,
        "global_distance": global_distance,
        "base_cluster_id": cluster_id,
        "base_top1_distance": top1_distance,
        "base_distance_band": distance_band,
        "base_cluster_persona_summary": cluster_persona_summary,
        "base_neighbors": base_neighbors or [],
        "base_vec_index": base_vec_index,
        "base_top_k_clusters": base_top_k_clusters or [],
        "base_persona": base_persona,
    }
    try:
        resp = await client.post(f"{CONTROLLER_URL}/detect", json=payload, timeout=PLANNER_TIMEOUT)
        resp.raise_for_status()
        planner_calls_total.labels(status="success").inc()
        return resp.json()
    except httpx.TimeoutException as e:
        # PLANNER_TIMEOUT 안에 controller의 agent loop(여러 라운드 analyze
        # 포함)가 안 끝난 경우 — analyzer(GGUF) 추론 지연이 누적된 결과일
        # 가능성이 높다. controller.py 로그의 같은 request_id로 어느
        # round/step에서 멈췄는지 대조해볼 것.
        logger.error("controller_agent_loop_timeout", request_id=request_id,
                     timeout_s=PLANNER_TIMEOUT, error=str(e))
        planner_calls_total.labels(status="timeout").inc()
        return fallback
    except httpx.HTTPStatusError as e:
        logger.error("controller_agent_loop_http_error", request_id=request_id,
                     status_code=e.response.status_code, body=e.response.text[:500])
        planner_calls_total.labels(status="http_error").inc()
        return fallback
    except Exception as e:
        # connection refused 등 — controller pod가 재시작 중이었을 가능성.
        # `kubectl get pods -n elastic` RESTARTS로 대조.
        logger.error("controller_agent_loop_failed", request_id=request_id,
                     error_type=type(e).__name__, error=str(e))
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

    # ── [진단] request_id는 controller.py에 그대로 전달되므로, 이 request_id로
    # controller 로그(step_start/step_done/step_failed)와 orchestrator 로그를
    # grep 한 번에 상관지어 볼 수 있다.
    logger.info("pipeline_start", request_id=request_id,
                experiment_case=experiment_case, percolate_version=percolate_version)

    async with httpx.AsyncClient() as client:
        # ── 1) Deterministic pre-router ──
        global_dist = compute_global_distance(tc.embedding)
        track = decide_track(global_dist, fast_threshold=FAST_TRACK_DIST_THRESHOLD)
        track_total.labels(track=track).inc()
        logger.info("pipeline_track_decided", request_id=request_id,
                    track=track, global_distance=round(global_dist, 4))

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
                termination="fast_track",
                latency_ms={"total": (time.time() - start) * 1000},
                experiment_case=experiment_case, percolate_version=percolate_version,
            )

        # ── 3) deep_retrieve (base): router + retriever ──
        # orchestrator가 직접 수행하는 것은 여기까지다. analyzer 호출,
        # sufficiency 진단, follow-up tool(expand/reroute) 실행, loop 반복은
        # 전부 controller.py의 /detect(LangGraph StateGraph)가 담당한다.
        _t0 = time.time()
        try:
            r0 = await call_router(client, tc.embedding, experiment_case, percolate_version)
        except Exception as e:
            logger.error("pipeline_router_failed", request_id=request_id,
                         error_type=type(e).__name__, error=str(e),
                         elapsed_ms=round((time.time() - _t0) * 1000, 1))
            raise
        primary_cluster_id = r0["primary_cluster_id"]
        clusters = [c["cluster_id"] for c in r0.get("top_k_clusters", [])]

        _t0 = time.time()
        try:
            rr = await call_retriever(client, tc.embedding, clusters, r0["vec_index"])
        except Exception as e:
            logger.error("pipeline_retriever_failed", request_id=request_id,
                         error_type=type(e).__name__, error=str(e),
                         elapsed_ms=round((time.time() - _t0) * 1000, 1))
            raise
        top1 = rr.get("top1_distance", 999.0)
        persona = r0.get("persona")
        persona_summary = persona.get("description") if isinstance(persona, dict) else None
        lat["base_retrieve"] = (time.time() - start) * 1000
        logger.info("pipeline_base_retrieve_done", request_id=request_id,
                    cluster_id=primary_cluster_id, top1_distance=round(top1, 4),
                    n_neighbors=len(rr.get("results", [])), elapsed_ms=round(lat["base_retrieve"], 1))

        # ── 4) Agent loop 전체를 controller.py로 위임 ──
        # (analyze → sufficiency 진단 → plan → expand/reroute → loop, 전부
        #  controller의 LangGraph StateGraph 안에서 수행된다.)
        _t0 = time.time()
        agent_result = await call_controller_agent_loop(
            client, request_id=request_id, transaction=tx, embedding=tc.embedding,
            global_distance=global_dist, cluster_id=primary_cluster_id, top1_distance=top1,
            distance_band=coarse_band(_normalize_dist(top1)), cluster_persona_summary=persona_summary,
            base_neighbors=rr.get("results", []),
            base_vec_index=r0.get("vec_index"),
            base_top_k_clusters=r0.get("top_k_clusters", []),
            base_persona=persona,   # r0 primary persona dict (없으면 None)
        )
        lat["controller_agent_loop"] = (time.time() - _t0) * 1000
        # ── [진단] controller가 fallback을 탔는지 여기서 바로 보인다 —
        # agent_trace에 "controller_unavailable_fallback"이 있으면 실패,
        # planner_used=False인데 fallback 문자열이 없으면 그 외 원인.
        logger.info(
            "pipeline_controller_loop_done", request_id=request_id,
            planner_used=agent_result.get("planner_used", False),
            agent_trace=agent_result.get("agent_trace", []),
            classification=agent_result.get("classification"),
            elapsed_ms=round(lat["controller_agent_loop"], 1),
        )

        # ── 5) Finalize — controller 응답을 orchestrator의 PipelineResponse로 매핑 ──
        lat["total"] = (time.time() - start) * 1000
        final_classification = agent_result.get("classification", "NORMAL")
        confidence = agent_result.get("confidence")
        if confidence is None:
            # controller가 confidence를 못 채운 극단적 fallback 상황 대비
            confidence = int(round(distance_to_score(_normalize_dist(agent_result.get("top_1_distance", top1))) * 100))
        logger.info("pipeline_done", request_id=request_id,
                    classification=final_classification, confidence=confidence,
                    latency_ms=lat)
        return PipelineResponse(
            classification=final_classification,
            confidence=int(confidence),
            reasoning=agent_result.get("reasoning") or "no_reasoning",
            key_evidence=[
                f"track=deep",
                f"top1={agent_result.get('top_1_distance', top1):.3f}",
                f"planner_used={agent_result.get('planner_used', False)}",
                f"tool_execution_mode={agent_result.get('tool_execution_mode', TOOL_EXECUTION_MODE)}",
            ],
            final_verdict=final_classification,
            primary_cluster_id=agent_result.get("primary_cluster_id", primary_cluster_id),
            top_5_clusters=[ClusterCandidate(**{k: v for k, v in c.items() if k != "persona"})
                            for c in r0.get("top_k_clusters", [])],   # persona는 최종 응답에서 제외
            vec_index=r0["vec_index"],
            match_type=r0.get("match_type", "strict"),
            match_score=float(r0.get("match_score", 1.0)),
            top_1_distance=agent_result.get("top_1_distance", top1),
            top_k=len(clusters),
            track="deep",
            planner_used=agent_result.get("planner_used", False),
            agent_trace=["pre_router:deep", "deep_retrieve"] + agent_result.get("agent_trace", []),
            latency_ms=lat,
            agent_rounds=agent_result.get("agent_rounds", []),
            termination=agent_result.get("termination"),
            experiment_case=experiment_case, percolate_version=percolate_version,
        )


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "orchestrator-agentic",
            "actions": list(FOLLOWUP_ACTIONS),
            "note": "agent loop (analyze/plan/tool/loop) now delegated to controller.py /detect"}


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
                controller_url=CONTROLLER_URL or "(not set)")

@app.get("/ready")
async def ready():
    try:
        async with httpx.AsyncClient() as client:
            checks = [
                ("router",    f"{ROUTER_URL}/health"),
                ("retriever", f"{RETRIEVER_URL}/health"),
                # analyzer는 orchestrator가 더 이상 직접 호출하지 않지만
                # (controller.py가 호출), 인프라 전체 가용성을 한눈에 보기
                # 위해 참고용으로 헬스체크만 유지한다.
                ("analyzer",  f"{ANALYZER_URL}/health"),
            ]
            if CONTROLLER_URL:
                checks.append(("controller_agent_loop", f"{CONTROLLER_URL}/ready"))

            results = {}
            for svc, url in checks:
                try:
                    r = await client.get(url, timeout=2)
                    results[svc] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
                except Exception as e:
                    results[svc] = f"error: {e}"

            if not CONTROLLER_URL:
                results["controller_agent_loop"] = "skipped (distance-only fallback in detect)"

        all_ok = all(v == "ok" or v.startswith("skipped") for v in results.values())
        if not all_ok:
            raise HTTPException(status_code=503, detail=results)

        return {
            "status":  "ready",
            "version": "8.0.0-agentic-thin",
            "services": results,
            "configuration": {
                "fast_track_dist_threshold": FAST_TRACK_DIST_THRESHOLD,
                "tool_execution_mode":       TOOL_EXECUTION_MODE,
                "controller_url":            CONTROLLER_URL or "(not set)",
                "followup_actions":          list(FOLLOWUP_ACTIONS),
                "note": "boundary_zone/max_expansion_iter/w_analyzer moved to controller.py "
                        "(AGENT_MAX_ROUNDS, sufficiency-based decision — no longer used here)",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
import os, time, math, json, asyncio, uuid
from typing import Dict, List, Optional, Literal, Any

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from controller import PlanLRUCache, build_plan_key, fallback_plan, PlanResponse

ROUTER_URL = os.getenv("ROUTER_URL", "http://anomaly-router:80")
RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://anomaly-retriever:80")
ANALYZER_URL = os.getenv("ANALYZER_URL", "http://local-analyzer:80")
CONTROLLER_URL = os.getenv("CONTROLLER_URL", "")
CONTROLLER_TIMEOUT = int(os.getenv("CONTROLLER_TIMEOUT", "8"))
CONTROLLER_MAX_RETRIES = int(os.getenv("CONTROLLER_MAX_RETRIES", "2"))
CONTROLLER_RPS = float(os.getenv("CONTROLLER_RPS", "5"))
CONF_GAP_THRESHOLD = float(os.getenv("CONF_GAP_THRESHOLD", "0.25"))
MAX_EXPANSION_ITER = int(os.getenv("MAX_EXPANSION_ITER", "1"))
GLOBAL_CENTROID_PATH = os.getenv("GLOBAL_CENTROID_PATH", "/data/global_centroid.npy")
TRANSPORT_MODE = os.getenv("TRANSPORT_MODE", "rest")  # rest | mcp_wrapper
PLAN_CACHE_SIZE = int(os.getenv("PLAN_CACHE_SIZE", "256"))

app = FastAPI(title="Anomaly Orchestrator Agentic", version="6.0.0-agentic")

controller_calls = Counter("controller_calls_total", "controller calls", ["status"])
controller_latency = Histogram("controller_latency_seconds", "controller latency")
controller_cache_hits = Counter("controller_plan_cache_hits_total", "controller plan cache hits")
feedback_loop_triggered = Counter("feedback_loop_triggered_total", "feedback triggered")
feedback_iteration_total = Counter("feedback_iteration_total", "feedback iterations")

plan_cache = PlanLRUCache(max_size=PLAN_CACHE_SIZE)
global_centroid: Optional[np.ndarray] = None

class TestCase(BaseModel):
    purchase_value: float
    age: float
    sex: Literal["M", "F"]
    source: Literal["Direct", "Ads", "SEO"]
    browser: Literal["Chrome", "IE", "Safari", "FireFox", "Opera"]
    embedding: List[float] = Field(..., min_length=576, max_length=576)

class ClusterCandidate(BaseModel):
    cluster_id: int
    rank: int
    score: float

class PipelineResponse(BaseModel):
    classification: str
    confidence: int
    reasoning: Optional[str]
    key_evidence: List[str]
    final_verdict: Optional[str]
    primary_cluster_id: int
    top_5_clusters: List[ClusterCandidate]
    vec_index: str
    match_type: str
    match_score: float
    top_1_distance: float
    top_k: int
    page_fault_delta: int
    is_hot: bool
    skip_analyzer: bool
    latency_ms: Dict[str, float]
    experiment_case: str
    percolate_version: str
    persona_used: bool
    tree_rules_used: bool


def distance_to_score(distance: float, threshold: float = 0.5) -> float:
    x = (distance - threshold) * 5
    return round(1 / (1 + math.exp(-x)), 4)


def compute_global_distance(embedding: List[float]) -> float:
    if global_centroid is None:
        return min(2.0, math.sqrt(sum(v*v for v in embedding[:64])) / 8.0)
    v = np.array(embedding, dtype=np.float32)
    denom = (np.linalg.norm(v) * np.linalg.norm(global_centroid) + 1e-8)
    cos = float(np.dot(v, global_centroid) / denom)
    return float(1.0 - cos)


async def call_controller_plan(client: httpx.AsyncClient, signal: Dict[str, Any], system_state: Dict[str, Any]) -> PlanResponse:
    key = build_plan_key(distance_hint=signal["global_distance"], resource_degraded=system_state["resource_degraded"], cache_hot=False)
    cached = plan_cache.get(key)
    if cached:
        controller_cache_hits.inc()
        cached.source = "lru_cache"
        return cached

    t0 = time.time()
    if CONTROLLER_URL:
        payload = {
            "query_signal": signal,
            "system_state": system_state,
            "schema": {"track": ["fast", "deep"], "sub_agents": ["distance", "evidence", "persona"], "expansion_budget": "int<=1"}
        }
        err = None
        for _ in range(CONTROLLER_MAX_RETRIES):
            try:
                resp = await client.post(f"{CONTROLLER_URL}/plan", json=payload, timeout=CONTROLLER_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                plan = PlanResponse(
                    track=data.get("track", "deep"),
                    sub_agents=data.get("sub_agents", ["distance", "evidence", "persona"]),
                    expansion_enabled=bool(data.get("expansion_enabled", True)),
                    expansion_budget=min(1, int(data.get("expansion_budget", 1))),
                    tie_break_enabled=bool(data.get("tie_break_enabled", False)),
                    source="llm",
                )
                plan_cache.put(key, plan)
                controller_calls.labels(status="success").inc()
                controller_latency.observe(time.time() - t0)
                return plan
            except Exception as e:
                err = e
                await asyncio.sleep(0.05)
        controller_calls.labels(status="fallback").inc()
    plan = fallback_plan(distance_hint=signal["global_distance"], resource_degraded=system_state["resource_degraded"])
    plan_cache.put(key, plan)
    controller_latency.observe(time.time() - t0)
    return plan


async def call_router(client: httpx.AsyncClient, embedding: List[float]) -> Dict:
    return (await client.post(f"{ROUTER_URL}/route", json={"embedding": embedding}, timeout=15)).json()

async def call_retriever(client: httpx.AsyncClient, embedding: List[float], cluster_ids: List[int], vec_index: str) -> Dict:
    endpoint = "/retrieve" if TRANSPORT_MODE == "rest" else "/mcp/execute"
    if endpoint == "/retrieve":
        payload = {"embedding": embedding, "cluster_ids": cluster_ids, "vec_index": vec_index, "top_k": 5}
    else:
        payload = {"tool": "retriever.retrieve", "arguments": {"embedding": embedding, "cluster_ids": cluster_ids, "vec_index": vec_index, "top_k": 5}}
    out = (await client.post(f"{RETRIEVER_URL}{endpoint}", json=payload, timeout=20)).json()
    return out.get("result", out)

async def call_analyzer(client: httpx.AsyncClient, tx: Dict, cluster_id: int, results: List[Dict], top1_distance: float, persona: Optional[Dict]) -> Dict:
    body = {"test_transaction": tx, "cluster_id": cluster_id, "top_k_normal_results": results, "top1_distance": top1_distance}
    if persona is not None:
        body["persona"] = persona
    return (await client.post(f"{ANALYZER_URL}/analyze", json=body, timeout=180)).json()


@app.post("/detect", response_model=PipelineResponse)
async def detect(tc: TestCase):
    start = time.time()
    lat: Dict[str, float] = {}
    tx = {"purchase_value": tc.purchase_value, "age": tc.age, "sex": tc.sex, "source": tc.source, "browser": tc.browser}
    async with httpx.AsyncClient() as client:
        signal = {"global_distance": compute_global_distance(tc.embedding)}
        system_state = {"resource_degraded": False, "transport_mode": TRANSPORT_MODE}
        plan = await call_controller_plan(client, signal, system_state)

        if plan.track == "fast":
            score = distance_to_score(signal["global_distance"])
            total = (time.time() - start) * 1000
            return PipelineResponse(
                classification="ABNORMAL" if score >= 0.5 else "NORMAL", confidence=int(score*100), reasoning="fast_track", key_evidence=[f"global_distance={signal['global_distance']:.4f}"], final_verdict=None,
                primary_cluster_id=-1, top_5_clusters=[], vec_index="", match_type="fast", match_score=0.0,
                top_1_distance=signal["global_distance"], top_k=0, page_fault_delta=0, is_hot=False, skip_analyzer=True,
                latency_ms={"total": total}, experiment_case="agentic", percolate_version="v14", persona_used=False, tree_rules_used=False
            )

        r0 = await call_router(client, tc.embedding)
        clusters = [c["cluster_id"] for c in r0.get("top_k_clusters", [])]
        rr = await call_retriever(client, tc.embedding, clusters, r0["vec_index"])
        top1 = rr.get("top1_distance", 999.0)
        results = rr.get("results", [])

        ar = await call_analyzer(client, tx, r0["primary_cluster_id"], results, top1, r0.get("persona"))

        # confidence gap loop (max_iter=1)
        loop_count = 0
        r_conf = max(0.0, min(1.0, 1.0 - min(1.0, top1)))
        a_conf = max(0.0, min(1.0, float(ar.get("confidence", 50)) / 100.0))
        gap = abs(r_conf - a_conf)
        if plan.expansion_enabled and gap >= CONF_GAP_THRESHOLD and loop_count < MAX_EXPANSION_ITER:
            feedback_loop_triggered.inc()
            feedback_iteration_total.inc()
            loop_count += 1
            expanded = [c["cluster_id"] for c in r0.get("top_k_clusters", [])][:5]
            rr2 = await call_retriever(client, tc.embedding, expanded, r0["vec_index"])
            top1 = rr2.get("top1_distance", top1)
            results = rr2.get("results", results)
            ar = await call_analyzer(client, tx, r0["primary_cluster_id"], results, top1, r0.get("persona"))

        total = (time.time() - start) * 1000
        lat["total"] = total
        return PipelineResponse(
            classification=ar.get("classification", "NORMAL"), confidence=int(ar.get("confidence", 50)), reasoning=ar.get("reasoning"), key_evidence=ar.get("key_evidence", []) + [f"plan_source={plan.source}", f"confidence_gap={gap:.3f}"],
            final_verdict=ar.get("final_verdict"),
            primary_cluster_id=r0["primary_cluster_id"], top_5_clusters=[ClusterCandidate(**c) for c in r0.get("top_k_clusters", [])], vec_index=r0["vec_index"],
            match_type=r0.get("match_type", "strict"), match_score=r0.get("match_score", 1.0),
            top_1_distance=top1, top_k=len(results), page_fault_delta=rr.get("page_fault_delta", 0), is_hot=r0.get("is_hot", False), skip_analyzer=False,
            latency_ms=lat, experiment_case="agentic", percolate_version="v14", persona_used=ar.get("persona_used", False), tree_rules_used=ar.get("tree_rules_used", False)
        )

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "orchestrator-agentic", "transport_mode": TRANSPORT_MODE}

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

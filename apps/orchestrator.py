"""
orchestrator.py — Pipeline Orchestrator V5.0
==============================================

Base: orchestrator.py V4.0
  ✅ Circuit Breaker (pybreaker) — router / retriever / analyzer 3중 보호
  ✅ Retry with exponential backoff
  ✅ Request-level cache (embedding prefix + 거래 필드 해시)
  ✅ Prometheus 메트릭 (pipeline / service / cb state / cache)

From orchestrator(1).py 통합:
  ★ Router v5 response 반영
      - purity / support / leaf_id 제거
      - vec_index 추가 (router가 결정 → retriever에 직접 전달)
      - skip_analyzer: bool 반환 여부 확인
  ★ Retriever v5 interface 반영
      - request: {embedding, cluster_ids, vec_index, top_k}
      - response: {results, top1_distance, page_fault_delta, search_mode}
      - SearchResult: {es_doc_id, original_index, cluster_id, distance, score}
  ★ Analyzer v3.0 interface 반영
      - query_id / leaf_id / support 제거
      - top_k_normal_results: retriever results 그대로 전달
      - top1_distance: retriever top1_distance 전달
  ★ SKIP_ANALYZER 분기
      - true: Stage 3 (Analyzer) 스킵 → top1_distance 기반 anomaly score
      - retriever에 cluster_ids=[] 전달 (전체 인덱스 검색)
  ★ page_fault_delta Prometheus pass-through
  ★ distance_to_score 함수 (SKIP_ANALYZER 시 사용)

V5.0 PipelineResponse 변경:
  - purity / support / leaf_id 제거
  - vec_index 추가
  - page_fault_delta 추가
  - skip_analyzer 추가
  - top_1_distance 유지
"""

import os
import time
import uuid
import hashlib
import asyncio
import math
from typing import Dict, Optional, List, Literal, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import structlog
import httpx
from pybreaker import CircuitBreaker, CircuitBreakerError

# ===========================
# Configuration
# ===========================

ROUTER_URL    = os.getenv("ROUTER_URL",    "http://anomaly-router:80")
RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://anomaly-retriever:80")
ANALYZER_URL  = os.getenv("ANALYZER_URL",  "http://anomaly-analyzer:80")

EXPERIMENT_CASE           = os.getenv("EXPERIMENT_CASE",           "pca_64")
DEFAULT_PERCOLATE_VERSION = os.getenv("DEFAULT_PERCOLATE_VERSION",  "v2")

ROUTER_TIMEOUT    = int(os.getenv("ROUTER_TIMEOUT",    "5"))
RETRIEVER_TIMEOUT = int(os.getenv("RETRIEVER_TIMEOUT", "10"))
ANALYZER_TIMEOUT  = int(os.getenv("ANALYZER_TIMEOUT",  "300"))
TOTAL_TIMEOUT     = int(os.getenv("TOTAL_TIMEOUT",     "600"))

CB_FAILURE_THRESHOLD = int(os.getenv("CB_FAILURE_THRESHOLD", "10"))
CB_RECOVERY_TIMEOUT  = int(os.getenv("CB_RECOVERY_TIMEOUT",  "60"))

MAX_RETRIES          = int(os.getenv("MAX_RETRIES",          "5"))
RETRY_BACKOFF_FACTOR = int(os.getenv("RETRY_BACKOFF_FACTOR", "4"))
RETRY_INITIAL_WAIT   = int(os.getenv("RETRY_INITIAL_WAIT",   "2"))

ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
CACHE_TTL    = int(os.getenv("CACHE_TTL", "600"))

# ★ V5: SKIP_ANALYZER — Analyzer 스킵, top1_distance 기반 score 산출
SKIP_ANALYZER      = os.getenv("SKIP_ANALYZER", "false").lower() == "true"
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.5"))

VALID_EXPERIMENT_CASES = [
    "emb_vectors",
    "pca_32", "pca_64", "pca_128", "pca_256",
    "k50", "k100", "k200",
    "pca_64_k100", "pca_64_k200",
]
VALID_PERCOLATE_VERSIONS = ["v1", "v2", "v3", "v4", "v5","v6","v7","v8","v9","v10","v11","v12","v13","v11a","v11b","v11c","v14"]

# ===========================
# FastAPI & Logger
# ===========================

app    = FastAPI(title="Anomaly Orchestrator V5.0 - Embedding Vector Based", version="5.0.0")
logger = structlog.get_logger()

# ===========================
# Circuit Breakers
# ===========================

router_breaker    = CircuitBreaker(fail_max=CB_FAILURE_THRESHOLD, reset_timeout=CB_RECOVERY_TIMEOUT, name="router")
retriever_breaker = CircuitBreaker(fail_max=CB_FAILURE_THRESHOLD, reset_timeout=CB_RECOVERY_TIMEOUT, name="retriever")
analyzer_breaker  = CircuitBreaker(fail_max=CB_FAILURE_THRESHOLD, reset_timeout=CB_RECOVERY_TIMEOUT, name="analyzer")

# ===========================
# Prometheus Metrics
# ===========================

pipeline_requests_total = Counter(
    "pipeline_requests_total", "Total pipeline requests",
    ["status", "percolate_version", "experiment_case"]
)
pipeline_latency = Histogram(
    "pipeline_latency_seconds", "End-to-end pipeline latency",
    buckets=[1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
)
service_latency = Histogram(
    "service_latency_seconds", "Individual service latency",
    ["service"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
)
circuit_breaker_state = Gauge(
    "circuit_breaker_state", "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    ["service"]
)
cache_hits   = Counter("cache_hits_total",   "Cache hits")
cache_misses = Counter("cache_misses_total", "Cache misses")
retry_attempts = Counter(
    "retry_attempts_total", "Total retry attempts", ["service", "attempt"]
)
experiment_case_counter = Counter(
    "experiment_case_requests_total", "Requests per experiment case", ["experiment_case"]
)
validation_errors = Counter(
    "validation_errors_total", "Validation errors", ["field"]
)
multi_cluster_search = Counter(
    "multi_cluster_search_total", "Multi-cluster search requests"
)
skip_analyzer_hits = Counter(
    "skip_analyzer_total", "Requests with SKIP_ANALYZER path"
)
# ★ V5: page fault pass-through from retriever
page_fault_pass = Histogram(
    "orchestrator_pf_delta", "Page fault delta from retriever",
    buckets=[0, 1, 2, 5, 10, 50, 100]
)

# ===========================
# Cache
# ===========================

cache_store: Dict[str, tuple] = {}

def get_cache_key(test_case: "TestCase") -> str:
    """embedding 앞 32차원 + 거래 필드 해시."""
    emb_prefix = str(test_case.embedding[:32])
    fields_str  = f"{test_case.purchase_value}{test_case.age}{test_case.sex}{test_case.source}{test_case.browser}"
    return hashlib.md5((emb_prefix + fields_str).encode()).hexdigest()

def get_from_cache(key: str) -> Optional[Dict]:
    if not ENABLE_CACHE:
        return None
    if key in cache_store:
        result, ts = cache_store[key]
        if time.time() - ts < CACHE_TTL:
            cache_hits.inc()
            return result
        del cache_store[key]
    cache_misses.inc()
    return None

def set_to_cache(key: str, result: Dict):
    if ENABLE_CACHE:
        cache_store[key] = (result, time.time())

# ===========================
# Pydantic Models
# ===========================

class TestCase(BaseModel):
    """
    파이프라인 입력 모델.
    embedding: AnoLLM 576-dim. Router 내부에서 PCA/전처리 수행.
    """
    purchase_value: float = Field(description="구매 금액 (z-score 정규화)")
    age:            float = Field(description="나이 (z-score 정규화)")
    sex:            Literal["M", "F"]
    source:         Literal["Direct", "Ads", "SEO"]
    browser:        Literal["Chrome", "IE", "Safari", "FireFox", "Opera"]

    weekday_purchase: Optional[Literal[
        "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"
    ]] = None
    month_purchase: Optional[Literal[
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]] = None
    IP_country: Optional[str] = None

    embedding: List[float] = Field(
        ..., description="AnoLLM 576-dim 임베딩 벡터"
    )

    experiment_case: Optional[str] = Field(
        default=None,
        description=f"실험 케이스 Override. 가능: {VALID_EXPERIMENT_CASES}"
    )
    percolate_version: Optional[Literal["v1","v2","v3","v4","v5","v6","v7","v8","v9","v10"]] = Field(
        default=None, description="Percolate query 버전 Override"
    )

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v):
        if len(v) != 576:
            validation_errors.labels(field="embedding").inc()
            raise ValueError(f"embedding must be 576-dim, got {len(v)}")
        return v

    @field_validator("experiment_case")
    @classmethod
    def validate_experiment_case(cls, v):
        if v is not None and v not in VALID_EXPERIMENT_CASES:
            validation_errors.labels(field="experiment_case").inc()
            raise ValueError(f"Invalid experiment_case '{v}'. Valid: {VALID_EXPERIMENT_CASES}")
        return v

    class Config:
        extra = "ignore"


class ClusterCandidate(BaseModel):
    """★ V5: purity / support / leaf_id 제거 (router v5 일관성)"""
    cluster_id: int
    rank:       int
    score:      float


class PipelineResponse(BaseModel):
    # ── 분류 결과 ──────────────────────────────────────────────
    classification:  str
    confidence:      int
    reasoning:       Optional[str]   # SKIP_ANALYZER=true면 None
    key_evidence:    List[str]
    final_verdict:   Optional[str]   # SKIP_ANALYZER=true면 None

    # ── Router 결과 (v5) ───────────────────────────────────────
    primary_cluster_id: int
    top_5_clusters:     List[ClusterCandidate]
    vec_index:          str           # ★ 신규: router가 결정한 vec 인덱스
    match_type:         str
    match_score:        float

    # ── Retriever 결과 ─────────────────────────────────────────
    top_1_distance:   float
    top_k:            int
    page_fault_delta: int             # ★ 신규: retriever page fault delta

    # ── 메타데이터 ─────────────────────────────────────────────
    is_hot:            bool
    skip_analyzer:     bool           # ★ 신규
    latency_ms:        Dict[str, float]
    experiment_case:   str
    percolate_version: str
    persona_used:      bool
    tree_rules_used:   bool

# ===========================
# Utility
# ===========================

def distance_to_score(distance: float) -> float:
    """
    top1_distance → anomaly score (0~1).
    SKIP_ANALYZER=true 시 Analyzer 없이 score 산출.
    distance < 0.3: score ~0.2 (정상)
    distance > 0.7: score ~0.8 (이상)
    """
    x = (distance - DISTANCE_THRESHOLD) * 5
    return round(1 / (1 + math.exp(-x)), 4)

# ===========================
# HTTP Client with Retry
# ===========================

async def call_with_retry(
    client:       httpx.AsyncClient,
    method:       str,
    url:          str,
    timeout:      int,
    service_name: str = "unknown",
    **kwargs
) -> httpx.Response:
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.request(method=method, url=url, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except httpx.HTTPError as e:
            last_error = e
            if hasattr(e, "response") and e.response and e.response.status_code < 500:
                raise
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_INITIAL_WAIT * (RETRY_BACKOFF_FACTOR ** attempt)
                retry_attempts.labels(service=service_name, attempt=str(attempt + 1)).inc()
                logger.warning("retry", service=service_name, attempt=attempt + 1, wait=wait)
                await asyncio.sleep(wait)
            else:
                raise last_error
    raise last_error

# ===========================
# Service Call Functions
# ===========================

@router_breaker
async def call_router(
    client:            httpx.AsyncClient,
    embedding:         List[float],
    experiment_case:   str,
    percolate_version: str,
) -> Dict:
    """
    Router v5 호출.

    Request:  POST /route  body={"embedding": [...576-dim...]}
    Headers:  X-Router-Case, X-Router-Version

    Response (v5):
      primary_cluster_id, top_k_clusters [{cluster_id, rank, score}],
      vec_index, match_type, match_score, is_hot,
      persona, skip_analyzer, latency_ms
      ★ purity / support / leaf_id 없음
    """
    start = time.time()
    try:
        resp = await call_with_retry(
            client, "POST", f"{ROUTER_URL}/route",
            timeout=ROUTER_TIMEOUT,
            service_name="router",
            json={"embedding": embedding},
            headers={
                "Content-Type":     "application/json",
                "X-Router-Case":    experiment_case,
                "X-Router-Version": percolate_version,
            }
        )
        result = resp.json()
        service_latency.labels(service="router").observe(time.time() - start)
        logger.info(
            "router_success",
            primary_cluster_id=result.get("primary_cluster_id"),
            vec_index=result.get("vec_index"),
            top_k=[c["cluster_id"] for c in result.get("top_k_clusters", [])],
            match_type=result.get("match_type"),
        )
        return result
    except Exception as e:
        logger.error("router_failed", error=str(e))
        raise


@retriever_breaker
async def call_retriever(
    client:      httpx.AsyncClient,
    embedding:   List[float],
    cluster_ids: List[int],
    vec_index:   str,
    top_k:       int = 5,
) -> Dict:
    """
    Retriever v5 호출.

    Request:  POST /retrieve
    Body: {embedding (576-dim), cluster_ids, vec_index, top_k}
      - SKIP_ANALYZER=true: cluster_ids=[] → 전체 인덱스 순수 vec 검색

    Response:
      results [{es_doc_id, original_index, cluster_id, distance, score}],
      top1_distance, page_fault_delta, search_mode, skip_analyzer
    """
    start = time.time()
    try:
        resp = await call_with_retry(
            client, "POST", f"{RETRIEVER_URL}/retrieve",
            timeout=RETRIEVER_TIMEOUT,
            service_name="retriever",
            json={
                "embedding":   embedding,
                "cluster_ids": cluster_ids,
                "vec_index":   vec_index,
                "top_k":       top_k,
            },
            headers={"Content-Type": "application/json"},
        )
        result = resp.json()
        service_latency.labels(service="retriever").observe(time.time() - start)
        multi_cluster_search.inc()
        logger.info(
            "retriever_success",
            vec_index=vec_index,
            cluster_ids=cluster_ids,
            n_results=len(result.get("results", [])),
            top1_distance=result.get("top1_distance"),
            page_fault_delta=result.get("page_fault_delta", 0),
        )
        return result
    except Exception as e:
        logger.error("retriever_failed", error=str(e))
        raise


@analyzer_breaker
async def call_analyzer(
    client:               httpx.AsyncClient,
    test_transaction:     Dict,
    cluster_id:           int,
    top_k_normal_results: List[Dict],
    top1_distance:        float,
    experiment_case:      str,
    persona:              Optional[Dict] = None,
    tree_features:        Optional[Dict] = None,
) -> Dict:
    """
    Analyzer v3.0 호출.

    ★ V5: query_id / leaf_id / support 제거
    Body:
      test_transaction, cluster_id,
      top_k_normal_results [{es_doc_id, original_index, cluster_id, distance, score}],
      top1_distance (LLM 컨텍스트용),
      persona (Optional), tree_features (Optional)
    Headers: X-Experiment-Case
    """
    start = time.time()
    try:
        body: Dict = {
            "test_transaction":     test_transaction,
            "cluster_id":           cluster_id,
            "top_k_normal_results": top_k_normal_results,
            "top1_distance":        top1_distance,
        }
        if persona is not None:
            body["persona"] = persona
        if tree_features is not None:
            body["tree_features"] = tree_features

        resp = await call_with_retry(
            client, "POST", f"{ANALYZER_URL}/analyze",
            timeout=ANALYZER_TIMEOUT,
            service_name="analyzer",
            json=body,
            headers={
                "Content-Type":      "application/json",
                "X-Experiment-Case": experiment_case,
            }
        )
        result = resp.json()
        service_latency.labels(service="analyzer").observe(time.time() - start)
        logger.info(
            "analyzer_success",
            classification=result.get("classification"),
            confidence=result.get("confidence"),
            persona_used=result.get("persona_used"),
            tree_rules_used=result.get("tree_rules_used"),
        )
        return result
    except Exception as e:
        logger.error("analyzer_failed", error=str(e))
        raise

# ===========================
# Main Pipeline
# ===========================

@app.post("/detect", response_model=PipelineResponse)
async def detect_anomaly(test_case: TestCase):
    """
    이상 탐지 파이프라인 V5.

    Flow (SKIP_ANALYZER=false):
      1. Router  → embedding → top_k_clusters (cluster_id/rank/score) + vec_index + persona
      2. Retriever → embedding + cluster_ids + vec_index → results + top1_distance + page_fault_delta
      3. Analyzer → test_transaction + top_k_normal_results + top1_distance → classification

    Flow (SKIP_ANALYZER=true):
      1. Router  → (동일)
      2. Retriever → cluster_ids=[] (전체 인덱스 검색)
      3. Analyzer 스킵 → top1_distance 기반 anomaly score
    """
    pipeline_start = time.time()
    request_id     = str(uuid.uuid4())[:8]

    exp_case = test_case.experiment_case    or EXPERIMENT_CASE
    pv       = test_case.percolate_version  or DEFAULT_PERCOLATE_VERSION

    experiment_case_counter.labels(experiment_case=exp_case).inc()

    logger.info(
        "pipeline_start",
        request_id=request_id,
        experiment_case=exp_case,
        percolate_version=pv,
        skip_analyzer=SKIP_ANALYZER,
        embedding_dim=len(test_case.embedding),
    )

    # ── 원본 거래 필드 dict (Analyzer 전달용) ─────────────────
    tx_dict: Dict = {
        "purchase_value": test_case.purchase_value,
        "age":            test_case.age,
        "sex":            test_case.sex,
        "source":         test_case.source,
        "browser":        test_case.browser,
    }
    if test_case.weekday_purchase:
        tx_dict["weekday_purchase"] = test_case.weekday_purchase
    if test_case.month_purchase:
        tx_dict["month_purchase"] = test_case.month_purchase
    if test_case.IP_country:
        tx_dict["IP_country"] = test_case.IP_country

    # ── 캐시 확인 ──────────────────────────────────────────────
    cache_key     = get_cache_key(test_case)
    cached_result = get_from_cache(cache_key)
    if cached_result:
        logger.info("cache_hit", request_id=request_id)
        return PipelineResponse(**cached_result)

    latencies: Dict[str, float] = {}

    try:
        async with httpx.AsyncClient() as client:

            # ── Stage 1: Router ────────────────────────────────
            try:
                t = time.time()
                router_result = await call_router(
                    client,
                    embedding=test_case.embedding,
                    experiment_case=exp_case,
                    percolate_version=pv,
                )
                latencies["router"] = (time.time() - t) * 1000
            except CircuitBreakerError:
                pipeline_requests_total.labels(
                    status="circuit_open", percolate_version=pv, experiment_case=exp_case
                ).inc()
                raise HTTPException(status_code=503, detail="Router unavailable (circuit open)")

            # router v5 response 파싱 (purity/support/leaf_id 없음)
            primary_cluster_id = router_result["primary_cluster_id"]
            top_k_clusters_raw = router_result.get("top_k_clusters", [])
            vec_index          = router_result["vec_index"]
            is_hot             = router_result.get("is_hot", False)
            persona            = router_result.get("persona")   or None
            match_type         = router_result.get("match_type",  "strict")
            match_score        = router_result.get("match_score", 1.0)

            top_5_clusters = [
                ClusterCandidate(
                    cluster_id=c["cluster_id"],
                    rank=c["rank"],
                    score=c["score"],
                )
                for c in top_k_clusters_raw
            ]
            # SKIP_ANALYZER=true → cluster_ids=[] (retriever가 전체 검색)
            cluster_ids = [] if SKIP_ANALYZER else [c.cluster_id for c in top_5_clusters]

            # ── Stage 2: Retriever ─────────────────────────────
            try:
                t = time.time()
                retriever_result = await call_retriever(
                    client,
                    embedding=test_case.embedding,   # 원본 576-dim 그대로
                    cluster_ids=cluster_ids,
                    vec_index=vec_index,
                    top_k=5,
                )
                latencies["retriever"] = (time.time() - t) * 1000
            except CircuitBreakerError:
                pipeline_requests_total.labels(
                    status="circuit_open", percolate_version=pv, experiment_case=exp_case
                ).inc()
                raise HTTPException(status_code=503, detail="Retriever unavailable (circuit open)")

            results_raw  = retriever_result.get("results", [])
            top1_dist    = retriever_result.get("top1_distance",   float("inf"))
            pf_delta     = retriever_result.get("page_fault_delta", 0)

            page_fault_pass.observe(pf_delta)

            # ── SKIP_ANALYZER 분기 ────────────────────────────
            if SKIP_ANALYZER:
                skip_analyzer_hits.inc()
                score      = distance_to_score(top1_dist)
                is_anomaly = score >= 0.5
                total_ms   = (time.time() - pipeline_start) * 1000
                latencies["total"] = total_ms
                pipeline_latency.observe(total_ms / 1000)

                result = PipelineResponse(
                    classification="ABNORMAL" if is_anomaly else "NORMAL",
                    confidence=int(score * 100),
                    reasoning=None,
                    key_evidence=[
                        f"top1_distance={top1_dist:.4f}",
                        f"threshold={DISTANCE_THRESHOLD}",
                        f"anomaly_score={score:.4f}",
                    ],
                    final_verdict=None,
                    primary_cluster_id=primary_cluster_id,
                    top_5_clusters=top_5_clusters,
                    vec_index=vec_index,
                    match_type=match_type,
                    match_score=match_score,
                    top_1_distance=top1_dist,
                    top_k=len(results_raw),
                    page_fault_delta=pf_delta,
                    is_hot=is_hot,
                    skip_analyzer=True,
                    latency_ms=latencies,
                    experiment_case=exp_case,
                    percolate_version=pv,
                    persona_used=False,
                    tree_rules_used=False,
                )
                set_to_cache(cache_key, result.model_dump())
                pipeline_requests_total.labels(
                    status="success_skip", percolate_version=pv, experiment_case=exp_case
                ).inc()
                logger.info(
                    "pipeline_done_skip",
                    request_id=request_id,
                    score=score,
                    top1_dist=top1_dist,
                    pf_delta=pf_delta,
                    total_ms=total_ms,
                )
                return result

            # ── Stage 3: Analyzer ──────────────────────────────
            try:
                t = time.time()
                analyzer_result = await call_analyzer(
                    client,
                    test_transaction=tx_dict,
                    cluster_id=primary_cluster_id,
                    top_k_normal_results=results_raw,   # retriever v5 SearchResult 그대로
                    top1_distance=top1_dist,
                    experiment_case=exp_case,
                    persona=persona,
                    tree_features=None,                 # router v5에서 tree_features 제거됨
                )
                latencies["analyzer"] = (time.time() - t) * 1000
            except CircuitBreakerError:
                pipeline_requests_total.labels(
                    status="circuit_open", percolate_version=pv, experiment_case=exp_case
                ).inc()
                raise HTTPException(status_code=503, detail="Analyzer unavailable (circuit open)")

        total_ms = (time.time() - pipeline_start) * 1000
        latencies["total"] = total_ms

        result = PipelineResponse(
            classification=analyzer_result["classification"],
            confidence=analyzer_result["confidence"],
            reasoning=analyzer_result.get("reasoning"),
            key_evidence=analyzer_result.get("key_evidence",  []),
            final_verdict=analyzer_result.get("final_verdict"),
            primary_cluster_id=primary_cluster_id,
            top_5_clusters=top_5_clusters,
            vec_index=vec_index,
            match_type=match_type,
            match_score=match_score,
            top_1_distance=top1_dist,
            top_k=len(results_raw),
            page_fault_delta=pf_delta,
            is_hot=is_hot,
            skip_analyzer=False,
            latency_ms=latencies,
            experiment_case=exp_case,
            percolate_version=pv,
            persona_used=analyzer_result.get("persona_used",    False),
            tree_rules_used=analyzer_result.get("tree_rules_used", False),
        )

        set_to_cache(cache_key, result.model_dump())

        pipeline_requests_total.labels(
            status="success", percolate_version=pv, experiment_case=exp_case
        ).inc()
        pipeline_latency.observe(total_ms / 1000)

        logger.info(
            "pipeline_done",
            request_id=request_id,
            classification=result.classification,
            primary_cluster_id=primary_cluster_id,
            vec_index=vec_index,
            top_1_distance=top1_dist,
            pf_delta=pf_delta,
            experiment_case=exp_case,
            total_ms=total_ms,
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        pipeline_requests_total.labels(
            status="error", percolate_version=pv, experiment_case=exp_case
        ).inc()
        logger.error("pipeline_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

# ===========================
# Health / Ready / Metrics
# ===========================

@app.get("/health")
async def health():
    return {
        "status":          "healthy",
        "service":         "orchestrator",
        "version":         "5.0.0",
        "skip_analyzer":   SKIP_ANALYZER,
        "experiment_case": EXPERIMENT_CASE,
    }


@app.get("/ready")
async def ready():
    try:
        async with httpx.AsyncClient() as client:
            checks = [
                ("router",    f"{ROUTER_URL}/health"),
                ("retriever", f"{RETRIEVER_URL}/health"),
            ]
            if not SKIP_ANALYZER:
                checks.append(("analyzer", f"{ANALYZER_URL}/health"))

            results = {}
            for svc, url in checks:
                try:
                    r = await client.get(url, timeout=2)
                    results[svc] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
                except Exception as e:
                    results[svc] = f"error: {e}"

            if SKIP_ANALYZER:
                results["analyzer"] = "skipped"

        all_ok = all(v in ("ok", "skipped") for v in results.values())
        if not all_ok:
            raise HTTPException(status_code=503, detail=results)

        return {
            "status":   "ready",
            "version":  "5.0.0",
            "services": results,
            "configuration": {
                "experiment_case":           EXPERIMENT_CASE,
                "default_percolate_version": DEFAULT_PERCOLATE_VERSION,
                "embedding_dim":             576,
                "skip_analyzer":             SKIP_ANALYZER,
                "distance_threshold":        DISTANCE_THRESHOLD,
                "valid_experiment_cases":    VALID_EXPERIMENT_CASES,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/metrics")
async def metrics():
    state_map = {"closed": 0, "open": 1, "half_open": 2}
    for name, breaker in [
        ("router",    router_breaker),
        ("retriever", retriever_breaker),
        ("analyzer",  analyzer_breaker),
    ]:
        circuit_breaker_state.labels(service=name).set(
            state_map.get(breaker.current_state, 0)
        )
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.on_event("startup")
async def startup_event():
    logger.info(
        "orchestrator_started",
        version="5.0.0",
        experiment_case=EXPERIMENT_CASE,
        percolate_version=DEFAULT_PERCOLATE_VERSION,
        skip_analyzer=SKIP_ANALYZER,
        distance_threshold=DISTANCE_THRESHOLD,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

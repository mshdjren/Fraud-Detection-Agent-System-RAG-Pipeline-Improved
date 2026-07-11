# ===========================
# Orchestrator orchestrator.py - Pipeline Orchestration V2
# Router → Retriever → Analyzer
# V2 Percolate Query 지원
# ===========================

import os
import time
import hashlib
import asyncio
import numpy as np
from typing import Dict, Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import structlog
import httpx
from pybreaker import CircuitBreaker, CircuitBreakerError

# ===========================
# Configuration
# ===========================

# Service URLs
ROUTER_URL = os.getenv("ROUTER_URL", "http://anomaly-router:80")
RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://anomaly-retriever:80")
ANALYZER_URL = os.getenv("ANALYZER_URL", "http://anomaly-analyzer:80")

# Timeouts
ROUTER_TIMEOUT = int(os.getenv("ROUTER_TIMEOUT", "5"))
RETRIEVER_TIMEOUT = int(os.getenv("RETRIEVER_TIMEOUT", "10"))
ANALYZER_TIMEOUT = int(os.getenv("ANALYZER_TIMEOUT", "30"))
TOTAL_TIMEOUT = int(os.getenv("TOTAL_TIMEOUT", "60"))

# Circuit Breaker
CB_FAILURE_THRESHOLD = int(os.getenv("CB_FAILURE_THRESHOLD", "5"))
CB_RECOVERY_TIMEOUT = int(os.getenv("CB_RECOVERY_TIMEOUT", "30"))

# Retry
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF_FACTOR = int(os.getenv("RETRY_BACKOFF_FACTOR", "2"))

# Cache
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))

# ===========================
# FastAPI App
# ===========================

app = FastAPI(title="Anomaly Orchestrator V2", version="2.0.0")

logger = structlog.get_logger()

# ===========================
# Circuit Breakers (서비스별)
# ===========================

router_breaker = CircuitBreaker(
    fail_max=CB_FAILURE_THRESHOLD,
    reset_timeout=CB_RECOVERY_TIMEOUT,
    name="router"
)

retriever_breaker = CircuitBreaker(
    fail_max=CB_FAILURE_THRESHOLD,
    reset_timeout=CB_RECOVERY_TIMEOUT,
    name="retriever"
)

analyzer_breaker = CircuitBreaker(
    fail_max=CB_FAILURE_THRESHOLD,
    reset_timeout=CB_RECOVERY_TIMEOUT,
    name="analyzer"
)

# ===========================
# Prometheus Metrics
# ===========================

pipeline_requests_total = Counter(
    "pipeline_requests_total",
    "Total pipeline requests",
    ["status", "router_version"]  # ✅ router_version 추가
)

pipeline_latency = Histogram(
    "pipeline_latency_seconds",
    "End-to-end pipeline latency",
    buckets=[1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

service_latency = Histogram(
    "service_latency_seconds",
    "Individual service latency",
    ["service"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

circuit_breaker_state = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    ["service"]
)

cache_hits = Counter("cache_hits_total", "Cache hits")
cache_misses = Counter("cache_misses_total", "Cache misses")

# ===========================
# Simple In-Memory Cache
# ===========================

cache_store: Dict[str, tuple[Dict, float]] = {}

def get_cache_key(test_case: Dict) -> str:
    """Generate cache key from test case"""
    data_str = str(sorted(test_case.items()))
    return hashlib.md5(data_str.encode()).hexdigest()

def get_from_cache(key: str) -> Optional[Dict]:
    """Get from cache if not expired"""
    if not ENABLE_CACHE:
        return None
    
    if key in cache_store:
        result, timestamp = cache_store[key]
        if time.time() - timestamp < CACHE_TTL:
            cache_hits.inc()
            return result
        else:
            del cache_store[key]
    
    cache_misses.inc()
    return None

def set_to_cache(key: str, result: Dict):
    """Set to cache with TTL"""
    if ENABLE_CACHE:
        cache_store[key] = (result, time.time())

# ===========================
# Pydantic Models
# ===========================

class TestCase(BaseModel):
    """입력 테스트 케이스 - V1/V2 호환"""
    
    # Numerical fields (공통)
    purchase_value: float
    age: float
    
    # ✅ V1 형식 (One-hot encoding) - 하위 호환성
    sex_M: float = Field(default=0.0)
    sex_F: float = Field(default=0.0)
    source_Direct: float = Field(default=0.0)
    source_Ads: float = Field(default=0.0)
    source_SEO: float = Field(default=0.0)
    browser_Chrome: float = Field(default=0.0)
    browser_IE: float = Field(default=0.0)
    browser_Safari: float = Field(default=0.0)
    browser_FireFox: float = Field(default=0.0)
    
    # ✅ V2 형식 (Keyword) - 신규
    sex: Optional[str] = Field(default=None, description="M or F")
    source: Optional[str] = Field(default=None, description="Direct, Ads, or SEO")
    browser: Optional[str] = Field(default=None, description="Chrome, IE, Safari, FireFox")
    
    # ✅ Router 버전 선택
    router_version: str = Field(default="v2", description="v1 or v2")

class PipelineResponse(BaseModel):
    """전체 파이프라인 응답"""
    classification: str  # NORMAL or ABNORMAL
    confidence: int  # 0-100
    cluster_id: int
    tree_id: int
    purity: float
    reasoning: str
    key_evidence: List[str]
    final_verdict: str
    
    # Latency breakdown
    latency_ms: Dict[str, float] = Field(
        description="Latency breakdown",
        example={
            "router": 45.2,
            "retriever": 123.4,
            "analyzer": 2345.6,
            "total": 2514.2
        }
    )
    
    # Metadata
    is_hot: bool
    top_k: int
    router_version: str  # ✅ 추가

# ===========================
# V1 → V2 변환
# ===========================

def convert_v1_to_v2(test_case: TestCase) -> Dict:
    """
    V1 형식 → V2 형식 변환
    
    V1: sex_M=1.0, sex_F=0.0
    V2: sex="M"
    """
    v2_data = {
        "purchase_value": test_case.purchase_value,
        "age": test_case.age
    }
    
    # ✅ V2 필드 우선 사용
    if test_case.sex:
        v2_data["sex"] = test_case.sex
    elif test_case.sex_M > 0.5:
        v2_data["sex"] = "M"
    elif test_case.sex_F > 0.5:
        v2_data["sex"] = "F"
    
    if test_case.source:
        v2_data["source"] = test_case.source
    elif test_case.source_Direct > 0.5:
        v2_data["source"] = "Direct"
    elif test_case.source_Ads > 0.5:
        v2_data["source"] = "Ads"
    elif test_case.source_SEO > 0.5:
        v2_data["source"] = "SEO"
    
    if test_case.browser:
        v2_data["browser"] = test_case.browser
    elif test_case.browser_Chrome > 0.5:
        v2_data["browser"] = "Chrome"
    elif test_case.browser_IE > 0.5:
        v2_data["browser"] = "IE"
    elif test_case.browser_Safari > 0.5:
        v2_data["browser"] = "Safari"
    elif test_case.browser_FireFox > 0.5:
        v2_data["browser"] = "FireFox"
    
    return v2_data

# ===========================
# HTTP Client with Retry
# ===========================

async def call_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    timeout: int,
    **kwargs
) -> httpx.Response:
    """
    HTTP call with exponential backoff retry
    
    Args:
        client: httpx AsyncClient
        method: HTTP method
        url: Target URL
        timeout: Request timeout
        **kwargs: Additional request parameters
    
    Returns:
        Response object
    
    Raises:
        httpx.HTTPError: After all retries failed
    """
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.request(
                method=method,
                url=url,
                timeout=timeout,
                **kwargs
            )
            response.raise_for_status()
            return response
        
        except httpx.HTTPError as e:
            last_error = e
            
            # 5xx 에러만 재시도
            if hasattr(e, 'response') and e.response:
                if e.response.status_code < 500:
                    raise
            
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_BACKOFF_FACTOR ** attempt
                logger.warning(
                    "retry_attempt",
                    url=url,
                    attempt=attempt + 1,
                    wait_seconds=wait_time,
                    error=str(e)
                )
                await asyncio.sleep(wait_time)
            else:
                raise last_error
    
    raise last_error

# ===========================
# Service Call Functions
# ===========================

@router_breaker
async def call_router(
    client: httpx.AsyncClient, 
    test_case: Dict,
    router_version: str = "v2"  # ✅ 추가
) -> Dict:
    """
    Router 호출 (Circuit breaker 보호, V2 지원)
    
    Args:
        client: HTTP client
        test_case: Test transaction
        router_version: "v1" or "v2"
    
    Returns:
        Router response (cluster_id, tree_id, purity, ...)
    """
    start_time = time.time()
    
    try:
        # ✅ V2 Header 추가
        headers = {"Content-Type": "application/json"}
        if router_version == "v2":
            headers["X-Router-Index"] = "v2"
        
        response = await call_with_retry(
            client,
            method="POST",
            url=f"{ROUTER_URL}/route",
            timeout=ROUTER_TIMEOUT,
            json=test_case,
            headers=headers  # ✅ 추가
        )
        
        result = response.json()
        latency = (time.time() - start_time) * 1000
        
        service_latency.labels(service="router").observe(latency / 1000)
        
        logger.info(
            "router_success",
            cluster_id=result.get("cluster_id"),
            router_version=router_version,  # ✅ 로깅
            match_type=result.get("match_type"),  # ✅ V2 전용
            match_score=result.get("match_score"),  # ✅ V2 전용
            latency_ms=latency
        )
        
        return result
    
    except Exception as e:
        logger.error("router_failed", error=str(e), router_version=router_version)
        raise

@retriever_breaker
async def call_retriever(
    client: httpx.AsyncClient,
    cluster_id: int,
    test_vector: List[float],
    is_hot: bool,
    top_k: int = 5
) -> Dict:
    """
    Retriever 호출 (Circuit breaker 보호)
    
    Args:
        client: HTTP client
        cluster_id: Cluster ID from Router
        test_vector: Test transaction vector (384 dim)
        is_hot: Hot cluster flag
        top_k: Number of similar docs
    
    Returns:
        Retriever response (top_k_results, ...)
    """
    start_time = time.time()
    
    try:
        response = await call_with_retry(
            client,
            method="POST",
            url=f"{RETRIEVER_URL}/search",
            timeout=RETRIEVER_TIMEOUT,
            json={
                "cluster_id": cluster_id,
                "test_vector": test_vector,
                "is_hot": is_hot,
                "top_k": top_k
            }
        )
        
        result = response.json()
        latency = (time.time() - start_time) * 1000
        
        service_latency.labels(service="retriever").observe(latency / 1000)
        
        logger.info(
            "retriever_success",
            cluster_id=cluster_id,
            found=len(result.get("top_k_results", [])),
            latency_ms=latency
        )
        
        return result
    
    except Exception as e:
        logger.error("retriever_failed", error=str(e))
        raise

@analyzer_breaker
async def call_analyzer(
    client: httpx.AsyncClient,
    query_id: str,
    cluster_id: int,
    tree_id: int,
    tree_features: Dict,
    top_k_normal_transactions: List[Dict]
) -> Dict:
    """
    Analyzer 호출 (Circuit breaker 보호)
    
    Args:
        client: HTTP client
        query_id: Test query ID
        cluster_id: Cluster ID
        tree_id: Tree ID
        tree_features: Tree features from Router
        top_k_normal_transactions: Similar normal transactions from Retriever
    
    Returns:
        Analyzer response (classification, confidence, reasoning, ...)
    """
    start_time = time.time()
    
    try:
        response = await call_with_retry(
            client,
            method="POST",
            url=f"{ANALYZER_URL}/analyze",
            timeout=ANALYZER_TIMEOUT,
            json={
                "query_id": query_id,
                "cluster_id": cluster_id,
                "tree_id": tree_id,
                "tree_features": tree_features,
                "top_k_normal_transactions": top_k_normal_transactions
            }
        )
        
        result = response.json()
        latency = (time.time() - start_time) * 1000
        
        service_latency.labels(service="analyzer").observe(latency / 1000)
        
        logger.info(
            "analyzer_success",
            query_id=query_id,
            classification=result.get("classification"),
            confidence=result.get("confidence"),
            latency_ms=latency
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
    전체 이상 탐지 파이프라인 (V2 지원)
    
    Args:
        test_case: Test transaction (V1 or V2 format)
    
    Returns:
        Complete anomaly detection result
    """
    pipeline_start = time.time()
    
    # Generate query ID
    query_id = f"query_{int(time.time() * 1000)}"
    
    logger.info(
        "pipeline_started",
        query_id=query_id,
        router_version=test_case.router_version  # ✅ 추가
    )
    
    # ✅ V1 → V2 변환 (router_version이 v2인 경우)
    if test_case.router_version == "v2":
        router_input = convert_v1_to_v2(test_case)
    else:
        router_input = test_case.model_dump()
    
    # Cache check (변환된 데이터 기준)
    cache_key = get_cache_key(router_input)
    cached_result = get_from_cache(cache_key)
    
    if cached_result:
        logger.info("cache_hit", query_id=query_id)
        return PipelineResponse(**cached_result)
    
    latencies = {}
    
    try:
        async with httpx.AsyncClient() as client:
            # ===========================
            # Step 1: Router - Cluster Assignment
            # ===========================
            
            try:
                router_start = time.time()
                router_result = await call_router(
                    client, 
                    router_input,
                    router_version=test_case.router_version  # ✅ 추가
                )
                latencies["router"] = (time.time() - router_start) * 1000
                
                cluster_id = router_result["cluster_id"]
                tree_id = router_result["tree_id"]
                purity = router_result.get("purity", 0.0)
                is_hot = router_result.get("is_hot", False)
                tree_features = router_result.get("tree_features", {})
                
            except CircuitBreakerError:
                pipeline_requests_total.labels(
                    status="circuit_open",
                    router_version=test_case.router_version
                ).inc()
                raise HTTPException(
                    status_code=503,
                    detail="Router service unavailable (circuit open)"
                )
            
            # ===========================
            # Step 2: Retriever - Top-K Search
            # ===========================
            
            # TODO: 실제로는 test_case를 vector로 변환 필요
            # 여기서는 mock vector 사용
            test_vector = np.random.randn(384).tolist()
            
            try:
                retriever_start = time.time()
                retriever_result = await call_retriever(
                    client,
                    cluster_id=cluster_id,
                    test_vector=test_vector,
                    is_hot=is_hot,
                    top_k=5
                )
                latencies["retriever"] = (time.time() - retriever_start) * 1000
                
                top_k_results = retriever_result.get("top_k_results", [])
                
            except CircuitBreakerError:
                pipeline_requests_total.labels(
                    status="circuit_open",
                    router_version=test_case.router_version
                ).inc()
                raise HTTPException(
                    status_code=503,
                    detail="Retriever service unavailable (circuit open)"
                )
            
            # ===========================
            # Step 3: Analyzer - LLM Analysis
            # ===========================
            
            try:
                analyzer_start = time.time()
                analyzer_result = await call_analyzer(
                    client,
                    query_id=query_id,
                    cluster_id=cluster_id,
                    tree_id=tree_id,
                    tree_features=tree_features,
                    top_k_normal_transactions=top_k_results
                )
                latencies["analyzer"] = (time.time() - analyzer_start) * 1000
                
                classification = analyzer_result["classification"]
                confidence = analyzer_result["confidence"]
                reasoning = analyzer_result["reasoning"]
                key_evidence = analyzer_result.get("key_evidence", [])
                final_verdict = analyzer_result.get("final_verdict", "")
                
            except CircuitBreakerError:
                pipeline_requests_total.labels(
                    status="circuit_open",
                    router_version=test_case.router_version
                ).inc()
                raise HTTPException(
                    status_code=503,
                    detail="Analyzer service unavailable (circuit open)"
                )
        
        # ===========================
        # Build Response
        # ===========================
        
        total_latency = (time.time() - pipeline_start) * 1000
        latencies["total"] = total_latency
        
        result = PipelineResponse(
            classification=classification,
            confidence=confidence,
            cluster_id=cluster_id,
            tree_id=tree_id,
            purity=purity,
            reasoning=reasoning,
            key_evidence=key_evidence,
            final_verdict=final_verdict,
            latency_ms=latencies,
            is_hot=is_hot,
            top_k=len(top_k_results),
            router_version=test_case.router_version  # ✅ 추가
        )
        
        # Cache result
        set_to_cache(cache_key, result.model_dump())
        
        # Metrics
        pipeline_requests_total.labels(
            status="success",
            router_version=test_case.router_version
        ).inc()
        pipeline_latency.observe(total_latency / 1000)
        
        logger.info(
            "pipeline_completed",
            query_id=query_id,
            classification=classification,
            confidence=confidence,
            router_version=test_case.router_version,
            total_latency_ms=total_latency
        )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        pipeline_requests_total.labels(
            status="error",
            router_version=test_case.router_version
        ).inc()
        logger.error(
            "pipeline_failed",
            query_id=query_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {str(e)}"
        )

# ===========================
# Health & Ready Endpoints
# ===========================

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "orchestrator", "version": "2.0.0"}


@app.get("/ready")
async def ready():
    """
    Readiness probe - 3개 서비스 연결 확인
    """
    try:
        async with httpx.AsyncClient() as client:
            # Router health check
            router_resp = await client.get(f"{ROUTER_URL}/health", timeout=2)
            router_resp.raise_for_status()
            
            # Retriever health check
            retriever_resp = await client.get(f"{RETRIEVER_URL}/health", timeout=2)
            retriever_resp.raise_for_status()
            
            # Analyzer health check
            analyzer_resp = await client.get(f"{ANALYZER_URL}/health", timeout=2)
            analyzer_resp.raise_for_status()
        
        # Circuit breaker states
        cb_states = {
            "router": router_breaker.current_state,
            "retriever": retriever_breaker.current_state,
            "analyzer": analyzer_breaker.current_state
        }
        
        return {
            "status": "ready",
            "version": "2.0.0",
            "services": {
                "router": "healthy",
                "retriever": "healthy",
                "analyzer": "healthy"
            },
            "circuit_breakers": cb_states
        }
    
    except Exception as e:
        logger.error("readiness_check_failed", error=str(e))
        raise HTTPException(
            status_code=503,
            detail=f"Services not ready: {str(e)}"
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    # Update circuit breaker state metrics
    for name, breaker in [
        ("router", router_breaker),
        ("retriever", retriever_breaker),
        ("analyzer", analyzer_breaker)
    ]:
        state_map = {"closed": 0, "open": 1, "half_open": 2}
        circuit_breaker_state.labels(service=name).set(
            state_map.get(breaker.current_state, 0)
        )
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# ===========================
# Startup Event
# ===========================

@app.on_event("startup")
async def startup_event():
    logger.info(
        "orchestrator_service_starting",
        version="2.0.0",
        router_url=ROUTER_URL,
        retriever_url=RETRIEVER_URL,
        analyzer_url=ANALYZER_URL,
        cache_enabled=ENABLE_CACHE,
        default_router_version="v2"  # ✅ 추가
    )

# ===========================
# Main
# ===========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
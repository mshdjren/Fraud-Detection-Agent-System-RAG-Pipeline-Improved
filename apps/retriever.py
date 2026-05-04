"""
retriever.py — Vector Search Retriever (V5.2)
============================================
변경사항:
    [03.31. V5.2 변경사항]: ingest.py의 _routing 반영
    - def search 함수 : routing_param 정의 및 routing_param 필요한 각 함수들에 인자 전달
    - query에도 cluster_id 정의 및 client.search에도 전달해야

  ★ V5.1: /retrieve-batch 엔드포인트 추가
      - batch_inference.py call_retriever_batch() 지원
      - 입력: List[RetrieveRequest] (각 request 독립적으로 search() 호출)
      - 출력: List[RetrieveResponse | BatchErrorResponse]
      - 개별 실패는 error 필드로 표시, 전체 batch는 200 반환
  ★ SKIP_ANALYZER=true 시 cluster_id terms filter 완전 제거
      → 전체 인덱스 대상 순수 vec 검색 (bottleneck isolation)
      → cluster_ids 파라미터가 비어있어도 검색 수행
  ★ SEARCH_MODE: "script_score"(Exact NN) | "knn"(HNSW) 전환
      - script_score: vec index=False 환경 (gaussian_aug 인덱스)
      - knn:          vec index=True 환경 (HNSW 그래프 활성화 인덱스)
  ★ gaussian_aug 인덱스명 규칙 지원:
      fraud_ecom_aug_{mult}x_{pct}pct_{vec_type}_vec
      (router가 vec_index를 동적으로 결정하여 전달)
  ★ page fault 측정: /proc/vmstat pgmajfault delta → Prometheus Histogram
  ★ ES node stats: fielddata / query_cache bytes Gauge (10 req마다 갱신)
"""

import os
import time
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import Optional, List
from elasticsearch import Elasticsearch
import structlog
import asyncio

import faiss

logger = structlog.get_logger()

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
ES_URL           = os.getenv("ES_URL")
PASSWORD         = os.getenv("PASSWORD")
SEARCH_MODE = os.getenv("SEARCH_MODE", "exact")   # "exact" | "hnsw"
KNN_CANDIDATES   = int(os.getenv("KNN_NUM_CANDIDATES", "100"))
TOP_K            = int(os.getenv("TOP_K", "5"))
REQUEST_TIMEOUT  = int(os.getenv("ES_REQUEST_TIMEOUT", "30"))
EXPERIMENT_LABEL = os.getenv("EXPERIMENT_LABEL", "baseline")  # Prometheus label / locust 연동용

# ★ SKIP_ANALYZER=true: cluster_id filter 제거 → 전체 인덱스 순수 vec 검색
SKIP_ANALYZER = os.getenv("SKIP_ANALYZER", "false").lower() == "true"

client = Elasticsearch(
    [ES_URL], verify_certs=False, ssl_show_warn=False,
    basic_auth=("elastic", PASSWORD), request_timeout=REQUEST_TIMEOUT,
)
app = FastAPI(title="Vector Search Retriever V5")

# ──────────────────────────────────────────────
# Prometheus
# ──────────────────────────────────────────────
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

search_latency = Histogram(
    "retriever_search_latency_ms", "Search latency per request",
    ["search_mode", "index_name", "experiment_label"],
    buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000]
)
page_fault_delta = Histogram(
    "retriever_page_fault_delta", "Major page fault delta per request",
    ["index_name"],
    buckets=[0, 1, 2, 5, 10, 50, 100]
)

es_fielddata_bytes   = Gauge("retriever_es_fielddata_bytes",   "ES fielddata memory bytes")
es_query_cache_bytes = Gauge("retriever_es_query_cache_bytes", "ES query cache memory bytes")

es_jvm_heap_percent  = Gauge("retriever_es_jvm_heap_percent",  "ES JVM heap used percent (max across nodes)")

top1_distance = Histogram(
    "retriever_top1_distance", "Top-1 cosine distance distribution",
    ["index_name", "experiment_label"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
)
skip_analyzer_search_counter = Counter(
    "retriever_skip_analyzer_total", "retriever_skip_analyzer_total_count",  ['index_name', 'experiment_label'] 
)




# ──────────────────────────────────────────────
# Page Fault / ES Node Stats
# ──────────────────────────────────────────────
def read_pgmajfault() -> int:
    """
    /proc/vmstat에서 pgmajfault(major page fault 누적값) 읽기.
    요청 전후 delta로 I/O 유발 메모리 압박 감지.
    """
    try:
        with open("/proc/vmstat") as f:
            for line in f:
                if line.startswith("pgmajfault"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0

_req_count = 0  # 10 req마다 node stats 갱신용

def update_es_node_stats():
    """
    ES node stats에서 fielddata / query_cache 메모리 추적.
    fielddata 폭증 → OOM 위험, query_cache 폭증 → GC pressure 시그널.
    10 req마다 호출 (매 요청마다 호출 시 ES 부하 증가).
    """
    try:
        stats = client.nodes.stats(metric="indices,jvm", index_metric="fielddata,query_cache")
        total_fd = total_qc = 0
        max_heap_pct = 0
        for node in stats["nodes"].values():
            total_fd += node["indices"]["fielddata"]["memory_size_in_bytes"]
            total_qc += node["indices"]["query_cache"]["memory_size_in_bytes"]
            # ★ coreset sweep: JVM heap percent (노드 중 최대값)
            heap_pct = node.get("jvm", {}).get("mem", {}).get("heap_used_percent", 0)
            max_heap_pct = max(max_heap_pct, heap_pct)
        es_fielddata_bytes.set(total_fd)
        es_query_cache_bytes.set(total_qc)
        es_jvm_heap_percent.set(max_heap_pct)  # ★
    except Exception:
        pass

# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────
class RetrieveRequest(BaseModel):
    embedding:   List[float]
    cluster_ids: List[int]        # router에서 전달받은 cluster_id 목록
                                  # SKIP_ANALYZER=true 시 빈 리스트([]) 허용 → 전체 검색
    vec_index:   str              # router가 결정한 인덱스명
    top_k:       int = 5


class SearchResult(BaseModel):
    es_doc_id:      str
    original_index: int
    cluster_id:     int
    distance:       float
    score:          float


class RetrieveResponse(BaseModel):
    results:         List[SearchResult]
    top1_distance:   float
    latency_ms:      float
    search_mode:     str
    page_fault_delta: int
    skip_analyzer:   bool

class BatchErrorResponse(BaseModel):
    """
    /retrieve-batch 단건 실패 시 반환.
    batch_inference._parse_retriever_response()가 r.get("error")로 판단.
    """
    error:   bool = True
    reason:  str  = ""
    results: list = []   # 빈 리스트 — _parse_retriever_response 호환


class McpExecuteRequest(BaseModel):
    tool: str
    arguments: dict


class McpExecuteResponse(BaseModel):
    ok: bool
    tool: str
    result: dict

# ──────────────────────────────────────────────
# Query Builders
# ──────────────────────────────────────────────
def _exact_knn_query(embedding: List[float], cluster_ids: List[int], k: int) -> dict:
    """
    Case 1 (gaussian aug): cluster filter + HNSW 탐색
    Case 3: 일반 coreset — cluster filter 후 전수 l2norm 계산 (Exact KNN).
    vec index=False 환경 전용.
    """
    if cluster_ids:
        filter_query = {"terms": {"cluster_id": cluster_ids}}
    else:
        # SKIP_ANALYZER=true: 전체 문서 대상 순수 vec 검색
        filter_query = {"match_all": {}}

    return {
        "size": k,
        "query": {
            "script_score": {
                "query": filter_query,
                "script": {
                    # "source": "cosineSimilarity(params.query_vector, 'vec') + 1.0",
                    "source": "1 / (1 + l2norm(params.query_vector, 'vec'))",
                    "params": {"query_vector": embedding}
                }
            }
        },
        "_source": ["original_index", "cluster_id"],
    }


def _hnsw_query(embedding: List[float], cluster_ids: List[int], k: int) -> dict:
    """
    Case 1/2: HNSW Approximate KNN — vec index=True 환경.
    Case 2 (SKIP_ANALYZER): filter 없이 전체 인덱스 HNSW 탐색 (baseline)
    """
    
    body: dict = {
        "size": k,
        "knn": {
            "field":           "vec",
            "query_vector":    embedding,
            "k":               k,
            "num_candidates":  KNN_CANDIDATES,
        },
        "_source": ["original_index", "cluster_id"],
    }

    # cluster_ids 비면 filter key 자체 생략 → 전체 인덱스 HNSW 탐색

    return body

def _hnsw_with_filter_query(embedding: List[float], cluster_ids: List[int], k: int) -> dict:
    """
    ES 8.x 이상에서 권장하는 'Filter 내장형 HNSW' (Pre-filtering)
    탐색 효율(logN)과 라우팅 정확도를 동시에 확보하는 가장 적합한 방식
    """
    body = {
        "size": k,
        "knn": {
            "field": "vec",
            "query_vector": embedding,
            "k": k,
            "num_candidates": KNN_CANDIDATES,
            # ★ 핵심: HNSW 그래프 탐색 시 클러스터 ID 조건을 필터로 내장
            "filter": {
                "terms": {"cluster_id": cluster_ids}
            } if cluster_ids else None
        },
        "_source": ["original_index", "cluster_id"],
    }
    # None인 필드는 삭제하여 전달
    if not body["knn"]["filter"]:
        del body["knn"]["filter"]
        
    return body


def _faiss_knn_query(embedding: List[float], cluster_ids: List[int], vec_index: str, k: int, routing_param: str) -> dict:
    """
    Step 1: ES에서 cluster_id 필터링된 후보 벡터(vec)와 ID를 Fetch
    """
    # 576차원 벡터 전수를 가져오기 위해 size를 넉넉히 설정 (필터링된 클러스터 내 데이터 수)
    candidate_query = {
        "size": 10000, 
        "query": {"terms": {"cluster_id": cluster_ids}} if cluster_ids else {"match_all": {}},
        "_source": ["vec", "original_index", "cluster_id"]
    }
    
    res = client.search(index=vec_index, body=candidate_query, routing=routing_param)
    hits = res["hits"]["hits"]
    
    if not hits:
        return []

    # Step 2: FAISS 전용 행렬(Numpy)로 변환
    node_vectors = np.array([hit["_source"]["vec"] for hit in hits]).astype('float32')
    query_vector = np.array([embedding]).astype('float32')

    # Step 3: FAISS Exact KNN (IndexFlatL2) 수행
    # SIMD 최적화가 적용된 물리적 연산 구간
    dim = len(embedding)
    index = faiss.IndexFlatL2(dim)
    index.add(node_vectors)
    distances, indices = index.search(query_vector, k)

    # Step 4: 결과를 ES 결과 형식과 호환되도록 역산 및 재구성
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue
        hit = hits[idx]
        d = float(distances[0][i])
        # ES score와 유사하게 1/(1+d) 형태로 변환 (선택 사항)
        score = 1.0 / (1.0 + d) 
        results.append(SearchResult(
            es_doc_id=hit["_id"],
            original_index=int(hit["_source"]["original_index"]),
            cluster_id=int(hit["_source"]["cluster_id"]),
            distance=np.sqrt(d), # L2 distance
            score=score
        ))
    return results

def _parse_es_hits(hits: List[dict]) -> List[SearchResult]:
    """ES 검색 결과를 공통 SearchResult 객체 리스트로 파싱 (L2 거리 역산 포함)"""
    parsed_results = []
    for hit in hits:
        src   = hit["_source"]
        score = hit["_score"]
        # ES l2_norm score = 1 / (1 + distance^2) 역산
        dist = np.sqrt((1.0 / score) - 1.0) if score > 0 else 1e6

        parsed_results.append(SearchResult(
            es_doc_id=      hit["_id"],
            original_index= int(src.get("original_index", -1)),
            cluster_id=     int(src.get("cluster_id", -1)),
            distance=       dist,
            score=          score,
        ))
    return parsed_results

# ──────────────────────────────────────────────
# Core Search (온전한 search 함수 재작성)
# ──────────────────────────────────────────────
def search(
    embedding:   List[float],
    cluster_ids: List[int],
    vec_index:   str,
    k:           int,
) -> RetrieveResponse:
    global _req_count

    # 분석기 스킵 여부 확인
    is_skip = SKIP_ANALYZER or not cluster_ids

    pf_before = read_pgmajfault()
    t0        = time.perf_counter()

    results = []

    routing_param = ",".join(map(str, cluster_ids)) if cluster_ids else None

    try:
        # ── 분기점 1: SKIP_ANALYZER (Case 2: Baseline 비교용) ──
        if is_skip:
            # cluster filter 없이 전체 인덱스 대상 HNSW 탐색
            body = _hnsw_query(embedding, [], k)
            res = client.search(index=vec_index, body=body)
            results = _parse_es_hits(res["hits"]["hits"])

        # ── 분기점 2: Case 1 & 3 (Cluster Filtering 기반 최적화 탐색) ──
        else:
            if SEARCH_MODE == "faiss":
                # Case 3-1: FAISS 기반 최적화 Exact KNN (ES 외부 연산)
                # 이 함수는 내부에서 검색과 SearchResult 구성을 모두 마쳐서 반환함
                results = _faiss_knn_query(embedding, cluster_ids, vec_index, k, routing_param)
            
            elif SEARCH_MODE == "hnsw_filtered":
                # Case 1-1: Filter 내장형 HNSW (Pre-filtering)
                body = _hnsw_with_filter_query(embedding, cluster_ids, k)
                res = client.search(index=vec_index, body=body, routing=routing_param)
                results = _parse_es_hits(res["hits"]["hits"])
            
            else:
                # Case 3-2: 기본 Exact KNN (script_score)
                # SEARCH_MODE가 "script_score"이거나 명시되지 않은 경우의 Fallback
                body = _exact_knn_query(embedding, cluster_ids, k)
                res = client.search(index=vec_index, body=body, routing=routing_param)
                results = _parse_es_hits(res["hits"]["hits"])

    except Exception as e:
        logger.error("search_failed", error=str(e), index=vec_index,
                     search_mode=SEARCH_MODE, skip_analyzer=is_skip)
        raise HTTPException(status_code=500, detail=str(e))

    elapsed  = (time.perf_counter() - t0) * 1000
    pf_delta = max(0, read_pgmajfault() - pf_before)

    # 상위 1건 거리 추출
    dist_top1 = results[0].distance if results else 1.0

    # ── Prometheus & Metrics 기록 ──
    search_latency.labels(
        search_mode=SEARCH_MODE,
        index_name=vec_index,
        experiment_label=EXPERIMENT_LABEL,
    ).observe(elapsed)

    page_fault_delta.labels(index_name=vec_index).observe(pf_delta)

    top1_distance.labels(
        index_name=vec_index,
        experiment_label=EXPERIMENT_LABEL,
    ).observe(dist_top1)

    if is_skip:
        skip_analyzer_search_counter.labels(
            index_name=vec_index,
            experiment_label=EXPERIMENT_LABEL
        ).inc()

    # 10 req마다 ES node stats 갱신
    _req_count += 1
    if _req_count % 10 == 0:
        update_es_node_stats()

    if pf_delta > 0:
        logger.warning("page_fault_detected",
                       delta=pf_delta, latency_ms=elapsed, index=vec_index)

    logger.info(
        "search_done",
        index=vec_index,
        search_mode=SEARCH_MODE,
        skip_analyzer=is_skip,
        cluster_ids_count=len(cluster_ids),
        results_count=len(results),
        top1_distance=dist_top1,
        latency_ms=elapsed,
    )

    return RetrieveResponse(
        results=         results,
        top1_distance=   dist_top1,
        latency_ms=      elapsed,
        search_mode=     SEARCH_MODE,
        page_fault_delta=pf_delta,
        skip_analyzer=   is_skip,
    )

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):    
    """
    vec_index: router가 결정한 인덱스명
      - 원본 coreset: fraud_ecom_percentage_{pct}_cluster_tree_vec
      - gaussian aug: fraud_ecom_aug_{mult}x_{pct}pct_{vec_type}_vec

    cluster_ids: router top-k cluster ID 목록
      - SKIP_ANALYZER=true 시 [] 전달 → 전체 인덱스 검색
    """
    return await asyncio.to_thread(
        search,
        req.embedding,
        req.cluster_ids,
        req.vec_index,
        req.top_k,
    )

@app.post("/retrieve-batch")
async def retrieve_batch(reqs: List[RetrieveRequest]):
    async def _run(req: RetrieveRequest):
        try:
            resp = await asyncio.to_thread(
                search, req.embedding, req.cluster_ids, req.vec_index, req.top_k
            )
            return resp.model_dump()
            # return resp.dict()
        except HTTPException as e:
            logger.warning(
                "retrieve_batch_item_failed",
                vec_index=req.vec_index,
                cluster_ids_count=len(req.cluster_ids),
                status=e.status_code,
                detail=e.detail,
            )
            return {"error": True, "reason": e.detail, "results": []}
        except Exception as e:
            logger.error("retrieve_batch_item_error", vec_index=req.vec_index, error=str(e))
            return {"error": True, "reason": str(e), "results": []}

    return await asyncio.gather(*[_run(req) for req in reqs])


@app.post("/mcp/execute", response_model=McpExecuteResponse)
async def mcp_execute(req: McpExecuteRequest):
    """
    Minimal MCP-style wrapper endpoint for retriever.
    Supported tool:
      - retriever.retrieve
    """
    if req.tool != "retriever.retrieve":
        raise HTTPException(status_code=400, detail=f"unsupported tool: {req.tool}")

    payload = RetrieveRequest(**req.arguments)
    resp = await retrieve(payload)
    return McpExecuteResponse(ok=True, tool=req.tool, result=resp.model_dump())

@app.get("/health")
def health():
    return {
        "status":          "healthy",
        "service":         "retriever",
        "search_mode":     SEARCH_MODE,
        "skip_analyzer":   SKIP_ANALYZER,
    }

@app.get("/ready")
async def ready():
    try:
        # Elasticsearch 클러스터 상태 확인 (router와 동일한 로직)
        info = client.cluster.health(timeout="5s")
        return {
            "status":        "ready",
            "elasticsearch": info.get("cluster_name"),
            "configuration": {
                "search_mode":   SEARCH_MODE,
                "skip_analyzer": SKIP_ANALYZER,
                "version":       "5.1.0"
            }
        }
    except Exception as e:
        logger.error("ready_check_failed", error=str(e))
        # 503 Service Unavailable 반환
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/metrics")
async def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ===========================
# Startup / Shutdown
# ===========================

@app.on_event("startup")
async def startup_event():
    logger.info(
        "retriever_started",
        version="5.1.0",
        search_mode=SEARCH_MODE,
        skip_analyzer=SKIP_ANALYZER
    )

@app.on_event("shutdown")
async def shutdown_event():
    client.close()
    logger.info("retriever_stopped")

# 이 블록은 항상 파일의 가장 마지막에 위치해야 합니다.
if __name__ == "__main__":
    import uvicorn
    # 여기 포트가 8000인지, 실제 서비스 포트와 맞는지 확인하세요.
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

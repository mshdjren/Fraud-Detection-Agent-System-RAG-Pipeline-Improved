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
import re
import time
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from elasticsearch import Elasticsearch
try:
    from elasticsearch import BadRequestError
except ImportError:                              # 구버전 클라이언트 대응
    from elasticsearch.exceptions import BadRequestError
import structlog
import asyncio

import faiss
import json
logger = structlog.get_logger()

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
ES_URL           = os.getenv("ES_URL")
PASSWORD         = os.getenv("PASSWORD")
SEARCH_MODE = os.getenv("SEARCH_MODE", "exact")   # "exact" | "hnsw"
KNN_CANDIDATES   = int(os.getenv("KNN_NUM_CANDIDATES", "100"))
# TOP_K            = int(os.getenv("TOP_K", "5"))
REQUEST_TIMEOUT  = int(os.getenv("ES_REQUEST_TIMEOUT", "30"))
EXPERIMENT_LABEL = os.getenv("EXPERIMENT_LABEL", "baseline")  # Prometheus label / locust 연동용

# ★ SKIP_ANALYZER=true: cluster_id filter 제거 → 전체 인덱스 순수 vec 검색
SKIP_ANALYZER = os.getenv("SKIP_ANALYZER", "false").lower() == "true"

RUN_ID = os.getenv("RUN_ID", EXPERIMENT_LABEL)
# ★ ES 8.18 rescore_vector — 양자화 후보를 원본 float32로 재채점
#   >1.0 일 때만 블록을 붙인다 (f32 인덱스에는 불필요)
RETRIEVER_OVERSAMPLE = float(os.getenv("RETRIEVER_OVERSAMPLE", "0"))

# ★ [GRPO Planner] contextual_knn의 soft should-boost 가중치. field_filters
#   (hard term filter, fetch_contrast용)와 달리 이 값은 knn 후보를 걸러내지
#   않고 점수만 가산한다 — embedding similarity는 항상 유지된 채 원본
#   feature context가 맞는 후보가 우선된다(인수인계 §1 "hybrid retrieval").
RETRIEVER_SHOULD_BOOST_WEIGHT = float(os.getenv("RETRIEVER_SHOULD_BOOST_WEIGHT", "40.0"))

# ── 이웃 feature 반환 (analyzer의 agrees/differs 라인에 필수) ──
#   미포함 시 prompts.format_retrieval_digest가 agrees/differs 줄을 통째로
#   생략하여 neighbor_insufficient 판정이 구조적으로 불가능해진다.
# _FEATURE_COLS = [c.strip() for c in os.getenv(
#     "RETRIEVER_FEATURE_COLS",
#     "purchase_value,age,sex,source,browser,IP_country,month_purchase,weekday_purchase"
# ).split(",") if c.strip()]
# ★ 6개 categorical + 2개 numeric 전부. 이게 짧으면 이웃 응답에 feature가 빠져
#   agrees/persona_match가 축소되고, planner의 candidate pool이 잘린다.
_FEATURE_COLS = ["purchase_value", "age", "sex", "source", "browser",
                 "IP_country", "month_purchase", "weekday_purchase"]

_SOURCE_FIELDS = ["original_index", "cluster_id"] + _FEATURE_COLS
FAISS_CANDIDATE_SIZE = int(os.getenv("FAISS_CANDIDATE_SIZE", "10000"))

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

es_index_store_bytes = Gauge(
    "retriever_es_index_store_bytes", "ES index store size bytes", ["index_name"])
es_index_segments = Gauge(
    "retriever_es_index_segments", "ES index segment count", ["index_name"])

top1_distance = Histogram(
    "retriever_top1_distance", "Top-1 raw L2 distance distribution",
    ["index_name", "experiment_label"],
    buckets=[0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0]
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
        istats = client.indices.stats(index="fraud_ecom_*", metric="store,segments")
        for _idx, _s in istats.get("indices", {}).items():
            es_index_store_bytes.labels(index_name=_idx).set(
                _s["total"]["store"]["size_in_bytes"])
            es_index_segments.labels(index_name=_idx).set(
                _s["total"]["segments"]["count"])
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
    top_k:       int = 3   # ★ 기본 inference top-k=3 (expand 시 controller가 5로 요청)
    field_filters: Optional[Dict[str, str]] = None   # ★ [V3] contrast용 축 고정 필터
                                                      #   {categorical_field: value} → term 필터
    # ★ [GRPO Planner] contextual_knn용 soft should-boost. field_filters(hard
    #   term filter)와 달리 후보를 걸러내지 않고 점수만 가산한다 — embedding
    #   similarity를 유지하면서 지정한 feature context가 맞는 후보를 우선시.
    #   {categorical_field: value} → bool.should term (minimum_should_match=0).
    should_boost: Optional[Dict[str, str]] = None
    # ★ [예약] attribute_knn 초기 설계의 raw term 절. 현재 dispatch는
    #   field_filters(dict)만 사용한다 — 값이 오면 무시하지 않고 경고를 남긴다
    #   (조용히 무시되던 과거 버그 유형 재발 방지).
    filter_clauses: Optional[List[Dict[str, Any]]] = None
    # ★ [B4 수정] 정의만 되고 search()로 전달되지 않던 필드. 이제 실제로
    #   전달되어, attribute_mode가 아니어도 support(필터 통과 문서 수)를
    #   계산해 응답에 싣는다.
    return_support:  bool = False

    # ★ [attribute_knn] field_filters(hard term)를 SEARCH_MODE와 무관하게
    #   반드시 적용시키는 스위치. 기존 dispatch는 SEARCH_MODE != "hnsw_filtered"
    #   이면 field_filters가 조용히 무시되는 경로로 새는데(인수인계에 기록된
    #   V2 corr=1.000 버그와 같은 유형), attribute_knn은 필터가 안 걸리면
    #   contextual/global과 동일해져 실험이 무의미해진다.
    attribute_mode: bool = False
    # aug 인덱스는 공백 컬럼명이 text로 색인돼 있어 '<field>.keyword'로만
    # term이 가능하다. 실패 시 자동으로 이 형태로 1회 재시도한다.
    attribute_keyword_fallback: bool = True

class SearchResult(BaseModel):
    es_doc_id:      str
    original_index: int
    cluster_id:     int
    distance:       float
    score:          float
    features:       Optional[Dict[str, Any]] = None   # ★ 이웃 원본 feature

class RetrieveResponse(BaseModel):
    results:         List[SearchResult]
    # ★ [B1 수정] 검색 0건일 때 1.0을 지어내지 않는다.
    #   기존 `if results else 1.0`은 거리 분포(P50=1.448, IQR=0.387)에서
    #   nd = sigmoid((1.0-1.448)/0.387) = 0.24 → "confident normal"이 되어,
    #   검색 실패가 "정상 거래"라는 조용한 오답으로 둔갑했다(routing 버그가
    #   500이 아니라 무증상으로 흘렀던 경로). None + zero_hit로 명시한다.
    top1_distance:   Optional[float] = None
    zero_hit:        bool = False           # ★ 검색 결과 0건 (호출자가 반드시 분기)
    latency_ms:      float
    search_mode:     str
    page_fault_delta: int
    skip_analyzer:   bool
    # ★ [B3 수정] 정의는 있었으나 생성자에 인자가 없어 항상 None이던 필드.
    #   의미도 교체: len(results)(= 항상 top_k, 정보량 0)가 아니라
    #   "필터/부스트 조건을 만족하는 인덱스 내 문서 수"(client.count).
    support:         Optional[int] = None
    # ★ [관측] soft rerank 전/후 top1 거리 — contextual_knn의 Δnd 부호 구조를
    #   데이터로 판정하기 위해 응답에 노출한다(기존엔 로그에만 있었다).
    top1_distance_pre_boost: Optional[float] = None
    view:            Optional[str] = None   # global | cluster | contextual | attribute
    features_used:   List[str] = []

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
        "_source": _SOURCE_FIELDS,
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
        "_source": _SOURCE_FIELDS,
    }

    # cluster_ids 비면 filter key 자체 생략 → 전체 인덱스 HNSW 탐색
    # ★ [EXP] ES 8.18 rescore — 양자화 인덱스에서 nd 캘리브레이션(P50/IQR) 보존
    if RETRIEVER_OVERSAMPLE > 1.0:
        body["knn"]["rescore_vector"] = {"oversample": RETRIEVER_OVERSAMPLE}
    return body

def _strip_unindexed_field(msg: str, should_boost, field_filters):
    """
    ES BadRequestError 메시지에서 'Cannot search on field [X] since it is
    not indexed nor has doc values' 패턴의 필드명을 뽑아 should_boost/
    field_filters에서 제거한다. 매핑에 doc_values가 없는 categorical
    필드(예: sex)가 섞여 들어왔을 때, 그 필드 하나 때문에 요청 전체를
    500으로 죽이지 않고 나머지 조건으로 재시도하기 위함.
    ★ 근본 수정은 아니다 — ES 인덱스 매핑에서 해당 필드에 keyword +
    doc_values(기본 true)를 켜는 게 정석이다. 이건 그때까지의 완화책.

    Returns: (new_should_boost, new_field_filters, stripped_field | None)
    """
    m = re.search(r"field \[(\w+)\]", msg or "")
    if not m:
        return should_boost, field_filters, None
    f = m.group(1)
    sb = {k: v for k, v in (should_boost or {}).items() if k != f} or None
    ff = {k: v for k, v in (field_filters or {}).items() if k != f} or None
    return sb, ff, f


def _hnsw_with_filter_query(embedding: List[float], cluster_ids: List[int], k: int
                            ,field_filters: Optional[Dict[str, str]] = None) -> dict:
    """
    ES 8.x 이상에서 권장하는 'Filter 내장형 HNSW' (Pre-filtering)
    탐색 효율(logN)과 라우팅 정확도를 동시에 확보하는 가장 적합한 방식

    ★ [재설계] should_boost는 더 이상 이 함수(=ES 쿼리)에 들어가지 않는다.
      이전 설계(top-level query.bool.should를 knn과 같은 body에 넣어 ES가
      점수를 더하게 하는 방식)는 `_score`가 순수 l2_norm 점수(1/(1+d²))라는
      _parse_es_hits의 전제를 깨뜨렸다 — should 매치가 있으면 score가 1.0을
      넘어갈 수 있고, 그러면 dist=sqrt(1/score-1)의 근호 안이 음수가 되어
      NaN이 나온다. 이 NaN이 SearchResult.distance에 실려 응답에 포함되면
      Starlette JSONResponse.render()가 json.dumps(..., allow_nan=False)로
      직렬화하다 500으로 죽는다 — 오라클 로그의 "Cannot search on field..."
      400(매핑 문제, 다른 버그)과는 별개로 실측된 새 버그다.
      → soft boost는 search()에서 순수 knn 결과를 더 넓게(pool) 받아온 뒤
      Python에서 재정렬하는 방식으로 옮겼다(_rerank_with_should_boost).
      이 함수는 이제 hard filter(cluster_ids/field_filters)만 담당한다.
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
        "_source": _SOURCE_FIELDS,
    }
    # ★ cluster_ids term + [V3] field_filters term을 bool.filter로 병합 (hard)
    _filters = []
    if cluster_ids:
        _filters.append({"terms": {"cluster_id": cluster_ids}})
    if field_filters:
        for _f, _v in field_filters.items():
            _filters.append({"term": {_f: _v}})   # categorical 축 고정 (같은 프로필)
    if _filters:
        body["knn"]["filter"] = {"bool": {"filter": _filters}}
    else:
        body["knn"].pop("filter", None)

    # ★ [EXP] ES 8.18 rescore — 양자화 인덱스에서 nd 캘리브레이션(P50/IQR) 보존
    if RETRIEVER_OVERSAMPLE > 1.0:
        body["knn"]["rescore_vector"] = {"oversample": RETRIEVER_OVERSAMPLE}
    return body


# ★ [GRPO Planner] should_boost 후보 pool 배율 — k*이 배수만큼 순수 knn으로
#   더 넓게 받아온 뒤 Python에서 재정렬한다. 좁게 받으면(=k만) boost가 원래
#   top-k 밖에 있던 "feature는 맞지만 조금 더 먼" 이웃을 끌어올 여지가 없어
#   should_boost가 사실상 무의미해진다.
SHOULD_BOOST_POOL_MULTIPLIER = int(os.getenv("SHOULD_BOOST_POOL_MULTIPLIER", "5"))


def _rerank_with_should_boost(
    results: List["SearchResult"], should_boost: Dict[str, str], k: int,
    weight: float = RETRIEVER_SHOULD_BOOST_WEIGHT,
) -> List["SearchResult"]:
    """
    순수 knn 결과(distance/score가 왜곡되지 않은 상태)를 받아, should_boost
    필드가 일치하는 후보를 우선시하도록 Python에서 재정렬한다. ES 쿼리
    레벨에서 점수를 섞지 않으므로 반환되는 SearchResult.distance/score는
    항상 진짜 knn 값 그대로다 — nd 계산(margin/reward)의 정확성이 boost
    적용 여부와 무관하게 보장된다.
    """
    if not should_boost:
        return results[:k]

    def _boost_score(r: "SearchResult") -> float:
        n_match = sum(1 for f, v in should_boost.items()
                      if str(r.features.get(f)) == str(v))
        return r.score + weight * n_match

    ranked = sorted(results, key=_boost_score, reverse=True)
    return ranked[:k]



def _faiss_knn_query(embedding: List[float], cluster_ids: List[int],
                     vec_index: str, k: int, routing_param: str) -> List[SearchResult]:
    """
    Step 1: ES에서 cluster_id 필터링된 후보 벡터(vec)와 ID를 Fetch

    ★ 전제: 인덱스 매핑에 `_source.excludes: ["vec"]`가 없어야 한다.
      ingest_tree_pipeline.gen_mapping_raw_vec()이 기본으로 vec를 _source에서
      제외하므로, 그 상태로 적재된 인덱스에서는 이 모드를 쓸 수 없다
      (쿼리의 _source 필터는 저장되지 않은 필드를 복원하지 못한다).
      → SEARCH_MODE=hnsw_filtered 를 쓰거나, exclude 없이 재적재할 것.
    """
    candidate_query = {
        "size": FAISS_CANDIDATE_SIZE,
        "query": {"terms": {"cluster_id": cluster_ids}} if cluster_ids else {"match_all": {}},
        "_source": _SOURCE_FIELDS + ["vec"],      # vec는 저장되어 있을 때만 반환됨
    }

    # ★ [FIX] routing이 빈 문자열/None이면 인자 자체를 빼야 한다 — search()의
    #   _es_search와 동일한 방어. 이 함수는 search() 바깥이라 헬퍼를 못 쓴다.
    res = (client.search(index=vec_index, body=candidate_query, routing=routing_param)
           if routing_param else client.search(index=vec_index, body=candidate_query))
    hits = res["hits"]["hits"]
    if not hits:
        return []

    # ★ [가드] Step 2 진입 전에 vec 존재를 확인 — 여기서 걸러야 KeyError가 안 난다
    if "vec" not in (hits[0].get("_source") or {}):
        logger.error("faiss_vec_unavailable", index=vec_index,
                     source_keys=sorted((hits[0].get("_source") or {}).keys()))
        raise HTTPException(
            status_code=501,
            detail=("SEARCH_MODE=faiss requires 'vec' in _source, but this index "
                    "excludes it (mapping _source.excludes). "
                    "Use SEARCH_MODE=hnsw_filtered, or re-ingest without the exclude."))

    # Step 2: FAISS 전용 행렬(Numpy)로 변환
    node_vectors = np.asarray([h["_source"]["vec"] for h in hits], dtype="float32")
    query_vector = np.asarray([embedding], dtype="float32")

    # Step 3: FAISS Exact KNN (IndexFlatL2) — 반환 distance는 squared L2
    dim = len(embedding)
    if node_vectors.shape[1] != dim:
        raise HTTPException(status_code=500,
            detail=f"vector dim mismatch: index={node_vectors.shape[1]} query={dim}")
    index = faiss.IndexFlatL2(dim)
    index.add(node_vectors)
    distances, indices = index.search(query_vector, min(k, len(hits)))

    # Step 4: SearchResult로 재구성
    #   score = 1/(1+d_sq) → _parse_es_hits의 역산 sqrt(1/score - 1) = L2 와 정합
    results: List[SearchResult] = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        hit = hits[int(idx)]
        # ★ aug 인덱스는 'IP country' 같은 공백 컬럼명으로 색인돼 있다. 여기서
        #   언더스코어로 통일해야 _FEATURE_COLS 조회가 성립한다.
        src = {str(_k).replace(" ", "_"): _v for _k, _v in hit["_source"].items()}
        d_sq = float(distances[0][i])
        results.append(SearchResult(
            es_doc_id=      hit["_id"],
            original_index= int(src.get("original_index", -1)),
            cluster_id=     int(src.get("cluster_id", -1)),
            distance=       float(np.sqrt(d_sq)),
            score=          1.0 / (1.0 + d_sq),
            features={_c: src.get(_c) for _c in _FEATURE_COLS   # ★ k 섀도잉 회피
                      if src.get(_c) is not None},
        ))
    return results

def _parse_es_hits(hits: List[dict]) -> List[SearchResult]:
    """ES 검색 결과를 공통 SearchResult 객체 리스트로 파싱 (L2 거리 역산 포함)"""
    parsed_results = []
    for hit in hits:
        src   = {str(_k).replace(" ", "_"): _v for _k, _v in hit["_source"].items()}
        score = hit["_score"]
        # ES l2_norm score = 1 / (1 + distance^2) 역산
        dist = np.sqrt((1.0 / score) - 1.0) if score > 0 else 1e6

        parsed_results.append(SearchResult(
            es_doc_id=      hit["_id"],
            original_index= int(src.get("original_index", -1)),
            cluster_id=     int(src.get("cluster_id", -1)),
            distance=       dist,
            score=          score,
            features=       {k: src.get(k) for k in _FEATURE_COLS if src.get(k) is not None},
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
    field_filters: Optional[Dict[str, str]] = None,   # ★ [V3] contrast 축 고정
    should_boost: Optional[Dict[str, str]] = None,    # ★ [GRPO] contextual_knn soft boost
    attribute_mode: bool = False,
    attribute_keyword_fallback: bool = True,
    return_support: bool = False,          # ★ [B4] 이제 실제로 전달된다
) -> RetrieveResponse:
    global _req_count

    # ★ [B5] vec_index 검증 — 빈 값이면 ES가 "전체 인덱스"로 해석해 조용히
    #   다른 인덱스(aug/다른 실험 case)까지 섞어 검색한다. 결과가 그럴듯하게
    #   나오므로 호출자는 알 수 없고, 거리/이웃/support가 전부 무의미해진다.
    #   (실측: vec_index="" 로 보내니 support가 단일 인덱스 sex:M 카운트(71501)를
    #    넘는 898673이 나왔다 — 여러 인덱스 합산)
    if not vec_index or not str(vec_index).strip():
        raise HTTPException(
            status_code=422,
            detail="vec_index is required — 빈 값은 ES에서 전체 인덱스 검색으로 "
                   "해석되어 조용히 잘못된 결과를 냅니다. router 응답의 "
                   "vec_index를 그대로 전달하세요.")

    # 분석기 스킵 여부 확인
    # ★ [GRPO] should_boost가 있으면 cluster_ids가 비어 있어도(global scope)
    #   is_skip 분기(=_hnsw_query, should_boost 미지원)로 새지 않고 반드시
    #   hnsw_filtered 경로(_hnsw_with_filter_query)를 타야 한다 — 그래야
    #   soft boost가 실제로 적용된다. 이전 로직은 cluster_ids=[]이면
    #   should_boost를 조용히 무시했다.
#   ★ attribute_mode도 동일하게 예외 처리해야 한다. attribute_knn은
    #     cluster_ids=[]로 호출되므로 `not cluster_ids`가 True가 되어 is_skip
    #     분기(_hnsw_query, filter 미지원)로 새고, term filter가 통째로 무시되어
    #     global_knn과 동일한 검색이 된다(과거 should_boost와 같은 함정).
    is_skip = ((SKIP_ANALYZER or not cluster_ids)
               and not should_boost
               and not (attribute_mode and field_filters))
    
    pf_before = read_pgmajfault()
    t0        = time.perf_counter()

    results = []

    # ★ [관측] 모든 분기가 동일 스키마로 채우는 진단 변수.
    #   분기마다 흩어져 있던 값을 한 곳에서 선언해, 어떤 경로로 나가든
    #   응답에 같은 필드가 실리도록 보장한다.
    _view: Optional[str] = None
    _features_used: List[str] = []
    _pre_boost_top1: Optional[float] = None
    _support: Optional[int] = None

    routing_param = ",".join(map(str, cluster_ids)) if cluster_ids else None
    _ROUTING_DEFAULT = object()   # sentinel — "routing_param을 그대로 쓴다"

    def _es_search(body: dict, index: str = None, routing=_ROUTING_DEFAULT):
        """★ routing이 빈 값이면 인자 자체를 빼야 한다.
        이 인덱스는 cluster_id로 _routing 색인돼 있어 ?routing= 이 나가면
        어떤 샤드에도 안 걸려 0건이 반환된다(ES 실측 확인:
        routing 없음 hits=5 / routing='' hits=0 / routing=93 hits=5).
        attribute_knn·global_knn은 cluster_ids=[]로 호출되므로 반드시 이 경로.

        routing 인자:
          - 생략(기본)  : routing_param이 truthy할 때만 붙인다 — 대부분의 호출부.
          - routing=None: routing_param이 있어도 **절대 붙이지 않는다**.
            is_skip 분기처럼 "cluster filter 없이 전역 탐색"이 목적인 곳은
            body에 cluster term이 없는데 routing만 붙으면 샤드가 조용히
            좁혀져 전역 탐색이 아니게 된다 — 그 경로는 반드시 이 값을 쓴다."""
        _idx = index or vec_index
        _r = routing_param if routing is _ROUTING_DEFAULT else routing
        if _r:
            return client.search(index=_idx, body=body, routing=_r)
        return client.search(index=_idx, body=body)

    def _count_support(terms: Dict[str, str], scoped: bool = False) -> Optional[int]:
        """★ [B3] 진짜 support — term 조건을 만족하는 인덱스 내 문서 수.
        /count 엔드포인트와 동일한 body를 쓴다(<10ms). len(results)는 항상
        top_k와 같아 정보량이 0이므로 그 값을 쓰지 않는다.
        scoped=True면 cluster_ids까지 AND — 지금은 전 tool이 global scope라
        기본 False."""
        if not terms:
            return None
        must = [{"term": {_k: _v}} for _k, _v in terms.items() if _v is not None]
        if scoped and cluster_ids:
            must.append({"terms": {"cluster_id": cluster_ids}})
        try:
            return int(client.count(index=vec_index,
                                    body={"query": {"bool": {"must": must}}})["count"])
        except Exception as _e:
            logger.warning("support_count_failed", terms=list(terms), error=str(_e))
            return None

    try:
        # ── 분기점 1: SKIP_ANALYZER (Case 2: Baseline 비교용) ──
        if is_skip:
            # cluster filter 없이 전체 인덱스 대상 HNSW 탐색
            body = _hnsw_query(embedding, [], k)
            # ★ 전역 탐색 — routing_param이 있어도 붙이면 안 된다(위 docstring 참고)
            res = _es_search(body, routing=None)
            results = _parse_es_hits(res["hits"]["hits"])
            _view = "global"

        # ── 분기점 2: Case 1 & 3 (Cluster Filtering 기반 최적화 탐색) ──
        else:
            # ★ attribute 후보마다 client.count가 나가 8만 회 ES 왕복이 발생했다.
            #   field_filters 조합은 실제로 수백 개뿐이라 캐시 히트율이 99%에 가깝다.
            _SUPPORT_CACHE: Dict[str, int] = {}
            _SUPPORT_CACHE_MAX = int(os.getenv("SUPPORT_CACHE_MAX", "50000"))
            if attribute_mode and field_filters:
                # ★ exact term filter로 부분집합을 만든 뒤 그 안에서 kNN.
                #   _hnsw_with_filter_query가 cluster_ids term + field_filters term을
                #   bool.filter로 병합하고, RETRIEVER_OVERSAMPLE>1이면 rescore도
                #   자동으로 붙인다(별도 ENABLED 플래그 없음).
                _ck = f"{vec_index}|" + "|".join(
                    f"{k}={v}" for k, v in sorted(field_filters.items()))
                if _ck in _SUPPORT_CACHE:
                    _sup = _SUPPORT_CACHE[_ck]
                else:
                    try:
                        _sup = int(client.count(index=vec_index, body={"query": {"bool": {
                            "filter": [{"term": {_f: _v}} for _f, _v in field_filters.items()
                                       if _v is not None]}}})["count"])
                    except Exception:
                        _sup = None
                    if _sup is not None and len(_SUPPORT_CACHE) < _SUPPORT_CACHE_MAX:
                        _SUPPORT_CACHE[_ck] = _sup
                # try:
                #     _sup = int(client.count(index=vec_index, body={"query": {"bool": {
                #         "filter": [{"term": {_f: _v}} for _f, _v in field_filters.items()
                #                    if _v is not None]}}})["count"])
                # except Exception:
                #     _sup = None
                # logger.info("attribute_branch", field_filters=field_filters,
                #             cluster_ids_count=len(cluster_ids or []), support=_sup, k=k)
                try:
                    body = _hnsw_with_filter_query(embedding, cluster_ids, k, field_filters)
                    _dbg = {**body, "knn": {**body["knn"],
                            "query_vector": f"<len={len(embedding or [])}>"}}
                    logger.info("attribute_body_exact", body=json.dumps(_dbg, ensure_ascii=False))
                    res = _es_search(body)
                    logger.info("attribute_raw_hits",
                                n=len(res["hits"]["hits"]),
                                total=(res["hits"]["total"] or {}).get("value"),
                                took=res.get("took"),
                                routing_used=bool(routing_param))
                except BadRequestError:
                    if not attribute_keyword_fallback:
                        raise
                    # aug 인덱스 대응 — 'IP country'는 text라 term 불가,
                    # '<field>.keyword' 서브필드로 1회 재시도
                    _ff = {(f"{_k.replace('_', ' ')}.keyword"
                            if _k in ("IP_country", "month_purchase", "weekday_purchase")
                            else _k): _v for _k, _v in field_filters.items()}
                    logger.warning("attribute_keyword_fallback", index=vec_index,
                                   fields=list(_ff.keys()))
                    body = _hnsw_with_filter_query(embedding, cluster_ids, k, _ff)
                    res = _es_search(body)
                
                # ★ attribute가 global과 실제로 다른 이웃을 가져오는가.
                #   support가 커서(수만) global 최근접이 필터 안에 그대로 들어오면
                #   두 tool이 같아진다 — 그 경우 조합 크기를 늘려야 축이 갈린다.
                # try:
                #     _g_body = _hnsw_query(embedding, [], k)
                #     _g_res = client.search(index=vec_index, body=_g_body)
                #     _g = _parse_es_hits(_g_res["hits"]["hits"])
                #     _a_ids = {r.original_index for r in results}
                #     _g_ids = {r.original_index for r in _g}
                #     logger.info("attribute_vs_global",
                #                 field_filters=field_filters, support=_sup,
                #                 jaccard=round(len(_a_ids & _g_ids)
                #                               / max(len(_a_ids | _g_ids), 1), 3),
                #                 top1_same=(bool(results) and bool(_g)
                #                            and results[0].original_index == _g[0].original_index),
                #                 d_attr=round(results[0].distance, 6) if results else None,
                #                 d_global=round(_g[0].distance, 6) if _g else None)
                # except Exception as _e:
                #     logger.warning("attribute_vs_global_failed", error=str(_e))
                
                results = _parse_es_hits(res["hits"]["hits"])
                # ★ [B3] dead variable(attr_support)이었던 것을 응답 경로에 연결.
                _support = _sup
                _view = "attribute"
                _features_used = list(field_filters.keys())

            elif should_boost:
                # ★ [버그 수정 — 재설계] should_boost는 SEARCH_MODE 설정과 무관하게
                #   항상 최우선으로 처리한다. 이전엔 ES 쿼리에 should를 직접 얹어
                #   _score를 오염시켰다(→ distance=NaN → JSONResponse allow_nan=False
                #   에서 500). 지금은 순수 knn/filter 쿼리로 k*POOL_MULTIPLIER만큼
                #   넓게 받아온 뒤(_score/거리 왜곡 없음) Python에서 재정렬한다.
                pool_k = max(k * SHOULD_BOOST_POOL_MULTIPLIER, k + 10)
                pool_k = min(pool_k, KNN_CANDIDATES)   # ES 제약: k <= num_candidates
                try:
                    body = _hnsw_with_filter_query(embedding, cluster_ids, pool_k, field_filters)
                    res = _es_search(body)
                except BadRequestError as e:
                    # ★ [완화책] 매핑에 doc_values 없는 필드 하나 때문에 전체
                    #   실패시키지 않고, 그 필드만 빼고 1회 재시도한다. should_boost는
                    #   더 이상 ES 쿼리에 안 들어가므로 field_filters만 대상.
                    _, ff2, bad_field = _strip_unindexed_field(str(e), None, field_filters)
                    if bad_field is None:
                        raise
                    logger.warning("field_filters_field_unmapped_retry", field=bad_field,
                                   index=vec_index, note="ES 매핑에 keyword+doc_values 설정 필요")
                    body = _hnsw_with_filter_query(embedding, cluster_ids, pool_k, ff2)
                    res = _es_search(body)
                pool_results = _parse_es_hits(res["hits"]["hits"])
                # ★ [관측] rerank 전 top1 = 순수 dense 최근접. 이 값과 rerank 후
                #   top1의 차이가 contextual_knn이 nd에 주는 영향의 전부다.
                _pre_boost_top1 = pool_results[0].distance if pool_results else None
                results = _rerank_with_should_boost(pool_results, should_boost, k)
                _view = "contextual"
                _features_used = list(should_boost.keys())
                _support = _count_support(should_boost)
                # ★ soft rerank가 실제로 순서를 바꿨는지 — 안 바뀌면 contextual이
                #   plain global HNSW와 동일해진다(과거 V2 corr=1.000 버그와 같은 증상)
                _pre = [r.original_index for r in pool_results[:k]]
                _post = [r.original_index for r in results[:k]]
                # ★ 재정렬이 top-1까지 도달하는지가 관건. nd는 top1 거리이므로
                #   순서만 바뀌고 top1이 그대로면 nd_after == nd_before가 되어
                #   contextual이 global과 구분되지 않는다.
                _pre_ids  = [r.original_index for r in pool_results[:k]]
                _post_ids = [r.original_index for r in results[:k]]
                _pre_all  = [r.original_index for r in pool_results]
                _rank_in_pool = (_pre_all.index(_post_ids[0])
                                 if _post_ids and _post_ids[0] in _pre_all else -1)
                _n_boost_hit = sum(
                    1 for r in pool_results
                    if r.features and all(str(r.features.get(_f)) == str(_v)
                                          for _f, _v in should_boost.items()))
                logger.info("contextual_branch", should_boost=should_boost,
                            pool_size=len(pool_results), k=k,
                            reordered=(_pre_ids != _post_ids),
                            top1_changed=(bool(_pre_ids) and bool(_post_ids)
                                          and _pre_ids[0] != _post_ids[0]),
                            jaccard_topk=round(
                                len(set(_pre_ids) & set(_post_ids))
                                / max(len(set(_pre_ids) | set(_post_ids)), 1), 3),
                            new_top1_rank_in_pool=_rank_in_pool,
                            boost_hits_in_pool=_n_boost_hit,
                            d_top1_pre=round(pool_results[0].distance, 6) if pool_results else None,
                            d_top1_post=round(results[0].distance, 6) if results else None)
            elif SEARCH_MODE == "faiss":
                # Case 3-1: FAISS 기반 최적화 Exact KNN (ES 외부 연산)
                # 이 함수는 내부에서 검색과 SearchResult 구성을 모두 마쳐서 반환함
                results = _faiss_knn_query(embedding, cluster_ids, vec_index, k, routing_param)
                _view = "cluster" if cluster_ids else "global"
            
            elif SEARCH_MODE == "hnsw_filtered":
                # Case 1-1: Filter 내장형 HNSW (Pre-filtering)
                try:
                    body = _hnsw_with_filter_query(embedding, cluster_ids, k, field_filters)
                    res = _es_search(body)
                except BadRequestError as e:
                    _, ff2, bad_field = _strip_unindexed_field(str(e), None, field_filters)
                    if bad_field is None:
                        raise
                    logger.warning("field_filters_field_unmapped_retry", field=bad_field,
                                   index=vec_index, note="ES 매핑에 keyword+doc_values 설정 필요")
                    body = _hnsw_with_filter_query(embedding, cluster_ids, k, ff2)
                    res = _es_search(body)
                results = _parse_es_hits(res["hits"]["hits"])
                _view = "cluster" if cluster_ids else "global"
                if field_filters:
                    _features_used = list(field_filters.keys())
                    _support = _count_support(field_filters)
            
            else:
                # Case 3-2: 기본 Exact KNN (script_score)
                # SEARCH_MODE가 "script_score"이거나 명시되지 않은 경우의 Fallback
                body = _exact_knn_query(embedding, cluster_ids, k)
                res = _es_search(body)
                results = _parse_es_hits(res["hits"]["hits"])
                _view = "cluster" if cluster_ids else "global"

    except Exception as e:
        logger.error("search_failed", error=str(e), index=vec_index,
                     search_mode=SEARCH_MODE, skip_analyzer=is_skip)
        raise HTTPException(status_code=500, detail=str(e))

    elapsed  = (time.perf_counter() - t0) * 1000
    pf_delta = max(0, read_pgmajfault() - pf_before)

    # ── ★ [B1 수정] 상위 1건 거리 — 0건일 때 값을 지어내지 않는다 ──
    #   기존 fallback 1.0은 nd=0.24("confident normal")로 해석되어, 검색
    #   실패가 정상 판정으로 둔갑했다. None을 반환하고 zero_hit를 세운다.
    _zero_hit = not results
    dist_top1 = results[0].distance if results else None
    if _zero_hit:
        logger.error("zero_hit_search", index=vec_index, search_mode=SEARCH_MODE,
                     view=_view, cluster_ids_count=len(cluster_ids or []),
                     field_filters=field_filters, should_boost=should_boost,
                     support=_support,
                     note="0건 반환 — 호출자는 nd를 새로 계산하지 말고 nd_before를 유지할 것")

    # ── Prometheus & Metrics 기록 ──
    search_latency.labels(
        search_mode=SEARCH_MODE,
        index_name=vec_index,
        experiment_label=EXPERIMENT_LABEL,
    ).observe(elapsed)

    page_fault_delta.labels(index_name=vec_index).observe(pf_delta)

    if dist_top1 is not None:            # ★ [B1] 0건이면 히스토그램을 오염시키지 않는다
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

    # ★ 각 분기가 results를 채웠는지 검증 — ES는 결과를 줬는데 파싱 누락으로
    #   조용히 빈 리스트가 반환되는 유형을 막는다(실측: attribute 분기에서 발생).
    if not results and locals().get("res") is not None:
        try:
            _n_raw = len(res["hits"]["hits"])
            if _n_raw:
                logger.error("results_lost_after_search", n_raw=_n_raw,
                             view=locals().get("_view"), mode=SEARCH_MODE,
                             hint="분기에 results = _parse_es_hits(...) 가 있는지 확인")
        except Exception:
            pass

    logger.info(
        "search_done",
        index=vec_index,
        search_mode=SEARCH_MODE,
        view=_view,
        support=_support,                 # ★ [B3] len(results)(=항상 top_k) 대신 진짜 support
        features_used=_features_used,
        skip_analyzer=is_skip,
        cluster_ids_count=len(cluster_ids),
        results_count=len(results),
        zero_hit=_zero_hit,
        top1_distance=dist_top1,
        top1_distance_pre_boost=_pre_boost_top1,
        latency_ms=elapsed,
    )
    

    # ★ [B3] return_support=True면 아직 support가 없는 경로에서도 계산한다.
    if return_support and _support is None:
        _support = _count_support(field_filters or should_boost or {})

    return RetrieveResponse(
        results=         results,
        top1_distance=   dist_top1,
        zero_hit=        _zero_hit,
        latency_ms=      elapsed,
        search_mode=     SEARCH_MODE,
        page_fault_delta=pf_delta,
        skip_analyzer=   is_skip,
        support=         _support,
        top1_distance_pre_boost=_pre_boost_top1,
        view=            _view,
        features_used=   _features_used,
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
    
    # ★ [B4] 조용히 무시되던 파라미터를 명시적으로 알린다.
    if req.filter_clauses:
        logger.warning("filter_clauses_ignored", n=len(req.filter_clauses),
                       note="현재 dispatch는 field_filters(dict)만 사용 — "
                            "raw term 절이 필요하면 field_filters로 변환해 보낼 것")
    if req.attribute_mode and req.should_boost:
        # ★ 두 모드가 동시에 오면 attribute(hard filter)가 이기고 should_boost는
        #   무시된다 — 조용히 넘어가면 contextual/attribute 비교 실험이 오염된다.
        logger.warning("should_boost_ignored_by_attribute_mode",
                       should_boost=list(req.should_boost),
                       field_filters=list(req.field_filters or {}))

    return await asyncio.to_thread(
        search,
        req.embedding,
        req.cluster_ids,
        req.vec_index,
        req.top_k,
        req.field_filters,      # ★ [V3] contrast 축 고정 필터
        req.should_boost,       # ★ [GRPO] contextual_knn soft boost
        req.attribute_mode,
        req.attribute_keyword_fallback,
        req.return_support,     # ★ [B4] 이제 실제로 전달된다
    )

@app.post("/retrieve-batch")
async def retrieve_batch(reqs: List[RetrieveRequest]):
    async def _run(req: RetrieveRequest):
        try:
            # ★ [B2 수정] field_filters/should_boost/attribute_mode가 전달되지
            #   않아, batch로 부르면 contextual_knn·attribute_knn이 조용히
            #   plain kNN으로 격하되던 버그. /retrieve와 동일 인자로 통일한다.
            resp = await asyncio.to_thread(
                search,
                req.embedding, req.cluster_ids, req.vec_index, req.top_k,
                req.field_filters, req.should_boost,
                req.attribute_mode, req.attribute_keyword_fallback,
                req.return_support,
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

class CountRequest(BaseModel):
    vec_index:     str
    field_filters: Dict[str, str]
    cluster_ids:   List[int] = []
    keyword_fallback: bool = True


@app.post("/count")
async def count(req: CountRequest):
    """★ attribute_knn의 support 사전 조회. term filter만 적용한 문서 수는
    매우 싸므로(<10ms), planner 프롬프트에 조합별 support를 미리 노출해
    support가 희박한 조합을 피하도록 학습시킬 수 있다."""
    def _body(ff: Dict[str, str]) -> dict:
        must = [{"term": {_k: _v}} for _k, _v in ff.items() if _v is not None]
        if req.cluster_ids:
            must.append({"terms": {"cluster_id": req.cluster_ids}})
        return {"query": {"bool": {"must": must}}}

    try:
        r = client.count(index=req.vec_index, body=_body(req.field_filters))
    except BadRequestError:
        if not req.keyword_fallback:
            raise
        _ff = {(f"{k.replace('_', ' ')}.keyword"
                if k in ("IP_country", "month_purchase", "weekday_purchase") else k): v
               for k, v in req.field_filters.items()}
        r = client.count(index=req.vec_index, body=_body(_ff))
    return {"count": int(r["count"])}

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
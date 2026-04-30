"""
Router Service - Embedding Vector (v5)
================================================

v5.0 변경사항:
  ★ query_id 완전 제거 (TestCase / RouteResponse)
  ★ purity / support / leaf_id ClusterCandidate 및 RouteResponse에서 제거
      - cluster_id / rank / score 만 반환
      - _source fetch: cluster_id, query, persona만 조회 (불필요 필드 제거)
  ★ SKIP_ANALYZER=true 경로 추가
      - percolate 완전 스킵 → 사전 등록된 cluster_id 목록 직접 반환
      - retriever가 cluster_id 필터 없이 전체 vec 검색 수행 (순수 vec 성능 측정)
      - ALL_CLUSTER_IDS env로 고정 목록 지정 가능 (없으면 인덱스에서 동적 로드)
  ★ vec_index 동적 결정: AUG_MULTIPLIER / AUG_PCT / VEC_TYPE env로 gaussian_aug 인덱스 선택
      - 기본: fraud_ecom_percentage_{pct}_cluster_tree_vec (원본 coreset)
      - aug: fraud_ecom_aug_{mult}x_{pct}pct_{vec_type}_vec

v4.0 유지:
  - EXPERIMENT_CASE 기반 인덱스 / 전처리
  - PCA 모델 startup 로드
  - Top-5 percolate + centroid fallback
      centroid: {GCS_TREE_BASE}/{EXPERIMENT_CASE}/cluster_centroids.npy shape=(K, D)
      D = vector_dim (실험 케이스 embedding 차원, PCA 후). 비교 벡터 = percolate_doc 공간
  - Hot/Cold 클러스터 추적
  - persona 반환
"""

import os
import time
import joblib
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Response, Header, Request
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from elasticsearch import Elasticsearch
import structlog
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import asyncio

logger = structlog.get_logger()

# ===========================
# Configuration
# ===========================

ES_URL               = os.getenv("ES_URL", "https://elasticsearch-ha-es-http:9200")
PASSWORD             = os.getenv("PASSWORD")
ES_REQUEST_TIMEOUT   = int(os.getenv("ES_REQUEST_TIMEOUT", "120"))

EXPERIMENT_CASE         = os.getenv("EXPERIMENT_CASE", "pca_64")
PERCOLATE_QUERY_VERSION = os.getenv("PERCOLATE_QUERY_VERSION", "v2")
INDEX_PERCOLATOR        = os.getenv(
    "INDEX_PERCOLATOR",
    f"fraud_ecom_{EXPERIMENT_CASE}_{PERCOLATE_QUERY_VERSION}_tree_rules_percolator"
)

PCA_MODEL_PATH       = os.getenv("PCA_MODEL_PATH", "/data/pca_model.joblib")
CENTROID_PATH        = os.getenv("CENTROID_PATH", "/data/cluster_centroids.npy")
GCS_TREE_BASE        = os.getenv("GCS_TREE_BASE",
                                  "gs://fraudecom/tree-search/tree_fraudecom")

# ★ v5.1: centroid GCS 경로용 K (클러스터 수)
# EXPERIMENT_CASE의 k{N} suffix에서 자동 파싱 (pca_64_k100 → 100, k200 → 200).
# suffix가 없는 케이스(pca_64, emb_vectors 등)는 N_CLUSTERS env로 명시 지정.
# GCS 경로: {GCS_TREE_BASE}/k{K}/cluster_centroids.npy  shape=(K, 576)
def _extract_k_from_case(case: str) -> Optional[int]:
    import re as _re
    m = _re.search(r"(?:^|_)k(\d+)$", case)
    return int(m.group(1)) if m else None

_n_clusters_env = os.getenv("N_CLUSTERS", "").strip()
N_CLUSTERS: Optional[int] = (
    int(_n_clusters_env) if _n_clusters_env
    else _extract_k_from_case(EXPERIMENT_CASE)
)

HOT_CLUSTER_PERCENTILE = int(os.getenv("CORESET_HOT_PERCENTILE", "70"))
PERSONA_ENABLED        = os.getenv("PERSONA_ENABLED", "true").lower() == "true"
CACHE_TTL_HOT          = int(os.getenv("CACHE_TTL_HOT", "3600"))
CACHE_TTL_COLD         = int(os.getenv("CACHE_TTL_COLD", "300"))

# ★ v5: SKIP_ANALYZER — percolate 스킵, 순수 vec 검색 성능 측정용
SKIP_ANALYZER = os.getenv("SKIP_ANALYZER", "false").lower() == "true"

# SKIP_ANALYZER=true 시 반환할 cluster_id 목록 (콤마 구분 정수)
# 미설정 시 percolator 인덱스에서 동적 로드
_ALL_CLUSTER_IDS_ENV = os.getenv("ALL_CLUSTER_IDS", "").strip()
_STATIC_CLUSTER_IDS: Optional[List[int]] = (
    [int(x) for x in _ALL_CLUSTER_IDS_ENV.split(",") if x.strip()]
    if _ALL_CLUSTER_IDS_ENV else None
)

# ★ v5: vec_index 동적 결정용 env
#   기본(원본 coreset):   AUG_MULTIPLIER 미설정
#   aug 인덱스:           AUG_MULTIPLIER=10, AUG_PCT=100, VEC_TYPE=float32
VEC_INDEX_CORESET_PCT = int(os.getenv("VEC_INDEX_CORESET_PCT", "100"))  # 100 | 10 | 1
AUG_MULTIPLIER        = os.getenv("AUG_MULTIPLIER", "").strip()          # "" | "2" | "5" | "10" ...
AUG_PCT               = int(os.getenv("AUG_PCT", "100"))
VEC_TYPE              = os.getenv("VEC_TYPE", "float32")                  # float32 | int8

# ===========================
# Experiment Case Config
# ===========================

EXPERIMENT_CONFIG = {
    "emb_vectors": {"vector_dim": 576, "use_pca": False},
    "pca_32":      {"vector_dim": 32,  "use_pca": True},
    "pca_64":      {"vector_dim": 64,  "use_pca": True},
    "pca_128":     {"vector_dim": 128, "use_pca": True},
    "pca_256":     {"vector_dim": 256, "use_pca": True},
    "k50":         {"vector_dim": 576, "use_pca": False},
    "k100":        {"vector_dim": 576, "use_pca": False},
    "k200":        {"vector_dim": 576, "use_pca": False},
    "pca_64_k100": {"vector_dim": 64,  "use_pca": True},
    "pca_64_k200": {"vector_dim": 64,  "use_pca": True},
}

EMB_INPUT_DIM         = 576
AVAILABLE_EXPERIMENTS = list(EXPERIMENT_CONFIG.keys())
AVAILABLE_VERSIONS = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                      "v9", "v10", "v11", "v11a", "v11b", "v11c",
                      "v12", "v13", "v14"]
VERSION_INDEX_ALIAS = {
    "v14":  "v1",
    "v11a": "v11",
    "v11b": "v11",
    "v11c": "v11",
}
# ===========================
# Global State
# ===========================

pca_model = None
centroid_matrix: Optional[np.ndarray] = None   # ★ (K, D) 클러스터 centroid 행렬
centroid_cluster_ids: Optional[List[int]] = None  # ★ centroid row i → cluster_id
cluster_frequency: Dict[int, int] = {}
last_reset_time = datetime.now()

# SKIP_ANALYZER=true 시 사용할 cluster_id 캐시
_cached_all_cluster_ids: Optional[List[int]] = None

# ★ Bucketization 전역 변수
BUCKET_DEPTH_PCT = os.getenv("BUCKET_DEPTH_PCT", "15")
ENABLE_BUCKETIZATION = os.getenv("ENABLE_BUCKETIZATION", "auto").lower()

bucket_config: Optional[Dict] = None
bucket_centers: Optional[Dict[str, np.ndarray]] = None  # 사전계산된 중심값

# ===========================
# Elasticsearch Client
# ===========================

client = Elasticsearch(
    [ES_URL],
    verify_certs=False,
    ssl_show_warn=False,
    basic_auth=("elastic", PASSWORD),
    request_timeout=ES_REQUEST_TIMEOUT,
)

# ===========================
# FastAPI & Prometheus
# ===========================

app = FastAPI(title="Anomaly Router Service v5.0 - Embedding Vector Based")

percolate_requests    = Counter("percolate_requests_total",      "Total percolate requests",       ["experiment_case", "version"])
percolate_latency     = Histogram("percolate_matching_time_ms",  "Percolate matching time ms",     ["experiment_case", "version"], buckets=[10, 25, 50, 100, 200, 500, 1000])
preprocessing_latency = Histogram("preprocessing_latency_ms",    "Preprocessing latency ms",       ["experiment_case"], buckets=[1, 5, 10, 25, 50, 100])
cluster_hits          = Counter("cluster_hits_total",             "Cluster hit count",              ["cluster_id", "experiment_case", "version"])
hot_clusters_count    = Gauge("hot_clusters_count",               "Number of hot clusters")
match_type_counter    = Counter("match_type_total",               "Match type distribution",        ["experiment_case", "version", "match_type"])
centroid_fallback_latency = Histogram(
    "centroid_fallback_latency_ms", "Centroid fallback latency ms",
    ["experiment_case", "version"], buckets=[1, 5, 10, 25, 50, 100, 200]
)
persona_delivered     = Counter("persona_delivered_total",        "Persona delivered count",        ["has_persona"])
# skip_analyzer_counter = Counter("skip_analyzer_requests_total",  "Requests via SKIP_ANALYZER path", [])
skip_analyzer_counter = Counter(
    'skip_analyzer_requests',
    'Skip analyzer requests count',
    ['experiment_case', 'cluster_id']  # ← label 정의
)


# ===========================
# Pydantic Models
# ===========================

class TestCase(BaseModel):
    """
    입력 거래 데이터.
    embedding: 576-dim AnoLLM 임베딩 (caller가 사전 계산하여 제공).
    raw 필드들은 선택 사항 (로깅/persona 컨텍스트용).
    """
    embedding: List[float] = Field(..., description="AnoLLM 576-dim embedding vector")

    purchase_value: Optional[float] = None
    age:            Optional[float] = None
    sex:            Optional[str]   = None
    source:         Optional[str]   = None
    browser:        Optional[str]   = None
    IP_country:     Optional[str]   = None

    class Config:
        extra = "ignore"


class ClusterCandidate(BaseModel):
    """
    ★ v5: purity / support / leaf_id 제거
    cluster_id / rank / score 만 반환
    """
    cluster_id: int
    rank:       int
    score:      float


class RouteResponse(BaseModel):
    """
    ★ v5: query_id / purity / support / leaf_id 제거
    vec_index 추가 (retriever가 검색할 인덱스명)
    """
    # Top-1
    primary_cluster_id: int

    # Top-5 후보 (cluster_id + rank + score)
    top_k_clusters: List[ClusterCandidate]

    # retriever에 전달할 vec 인덱스명
    vec_index: str

    # Metadata
    is_hot:          bool
    cache_ttl:       int
    latency_ms:      float
    index_used:      str
    experiment_case: str
    version:         str
    match_type:      str
    match_score:     float
    skip_analyzer:   bool

    # 추가 정보
    tree_features:       Dict[str, Any]
    persona:             Optional[Dict[str, Any]] = None
    preprocessing_stats: Optional[Dict[str, Any]] = None

# ===========================
# vec_index 결정
# ===========================

def get_vec_index() -> str:
    """
    ★ v5: vec_index 동적 결정
      - AUG_MULTIPLIER 설정 시: fraud_ecom_aug_{mult}x_{pct}pct_{vec_type}_vec
      - 미설정 시:            : fraud_ecom_{experiment_case}_percentage_{percentage}_cluster_tree_vec

    """
    if AUG_MULTIPLIER:
        return f"fraud_ecom_aug_{AUG_MULTIPLIER}x_{AUG_PCT}pct_{VEC_TYPE}_vec"
    return f"fraud_ecom_{EXPERIMENT_CASE}_percentage_{VEC_INDEX_CORESET_PCT}_cluster_tree_vec"

# ===========================
# Preprocessing
# ===========================

def preprocess_to_percolate_doc(
    embedding: List[float],
    experiment_case: str,
    pca_mdl: Optional[Any] = None,
) -> Dict[str, float]:
    """
    AnoLLM embedding → percolate document {"v1": ..., "v2": ..., ...}
    - non-PCA: embedding 그대로 v1~v576
    - PCA:     pca_model.transform(embedding) → v1~v{pca_dim}
    """
    config    = EXPERIMENT_CONFIG.get(experiment_case, {})
    use_pca   = config.get("use_pca", False)
    vector_dim = config.get("vector_dim", EMB_INPUT_DIM)

    vec = np.array(embedding, dtype=np.float32).reshape(1, -1)

    if use_pca:
        if pca_mdl is None:
            raise ValueError(
                f"PCA model required for experiment_case='{experiment_case}' but not loaded."
            )
        vec = pca_mdl.transform(vec).astype(np.float32)

    vec = vec.flatten()[:vector_dim]
    return {f"v{i+1}": float(v) for i, v in enumerate(vec)}

# ===========================
# Hot Cluster Tracking
# ===========================

def reset_frequency_if_needed():
    global last_reset_time, cluster_frequency
    if (datetime.now() - last_reset_time) > timedelta(hours=1):
        cluster_frequency = {}
        last_reset_time = datetime.now()

def update_cluster_frequency(cluster_id: int):
    reset_frequency_if_needed()
    cluster_frequency[cluster_id] = cluster_frequency.get(cluster_id, 0) + 1

def get_hot_clusters() -> List[int]:
    if not cluster_frequency:
        return []
    sorted_clusters = sorted(cluster_frequency.items(), key=lambda x: x[1], reverse=True)
    total = sum(cluster_frequency.values())
    target = total * (HOT_CLUSTER_PERCENTILE / 100.0)
    hot, cumulative = [], 0
    for cid, count in sorted_clusters:
        if cumulative >= target:
            break
        hot.append(cid)
        cumulative += count
    return hot

# ===========================
# Tree Features Extraction
# ===========================

def extract_tree_features(query: Dict[str, Any]) -> Dict[str, Any]:
    features = {}
    if "bool" not in query:
        return features
    bool_q = query["bool"]
    for clause_key in ("filter", "should", "must"):
        for cond in bool_q.get(clause_key, []):
            if "bool" in cond:
                for inner_cond in cond["bool"].get("filter", []):
                    _extract_range(inner_cond, features)
            else:
                _extract_range(cond, features)
    if "minimum_should_match" in bool_q:
        features["minimum_should_match"] = bool_q["minimum_should_match"]
    return features

def _extract_range(cond: Dict, features: Dict):
    if "range" not in cond:
        return
    field = list(cond["range"].keys())[0]
    rc    = cond["range"][field]
    if "lte" in rc or "lt" in rc:
        features[f"{field}_max"] = rc.get("lte", rc.get("lt"))
    if "gte" in rc or "gt" in rc:
        features[f"{field}_min"] = rc.get("gte", rc.get("gt"))


# ===========================
# SKIP_ANALYZER: cluster_id 목록 조회
# ===========================

def get_all_cluster_ids(percolator_index: str) -> List[int]:
    """
    ★ v5: SKIP_ANALYZER=true 시 percolate 스킵 → 전체 cluster_id 목록 반환.
    retriever가 cluster_id 필터 없이 전체 vec 검색 수행.

    - _STATIC_CLUSTER_IDS env 설정 시 해당 목록 고정 사용
    - 미설정 시 percolator 인덱스에서 고유 cluster_id 동적 로드 (startup 1회 캐시)
    """
    global _cached_all_cluster_ids

    if _STATIC_CLUSTER_IDS is not None:
        return _STATIC_CLUSTER_IDS

    if _cached_all_cluster_ids is not None:
        return _cached_all_cluster_ids

    try:
        result = client.search(
            index=percolator_index,
            body={
                "size": 0,
                "aggs": {
                    "unique_clusters": {
                        "terms": {"field": "cluster_id", "size": 10000}
                    }
                }
            }
        )
        cluster_ids = [
            int(b["key"])
            for b in result["aggregations"]["unique_clusters"]["buckets"]
        ]
        _cached_all_cluster_ids = cluster_ids
        logger.info("skip_analyzer_cluster_ids_loaded",
                    count=len(cluster_ids), index=percolator_index)
        return cluster_ids
    except Exception as e:
        logger.error("skip_analyzer_cluster_load_failed", error=str(e))
        return []

# 위치: Line 300 (Helper Functions 섹션)
# ===========================
# Bucketization
# ===========================

def calculate_bucket_indices(
    vector: np.ndarray,
    experiment_case: str
) -> Dict[str, int]:
    """입력 벡터의 각 feature별 bucket index 계산"""
    if not bucket_config:
        return {}
    
    indices = {}
    split_map = bucket_config["split_points"]
    
    # PCA 적용된 경우 vector는 이미 변환된 상태
    vec_dict = {f"v{i+1}": val for i, val in enumerate(vector)}
    
    for feature in bucket_config["split_points"].keys():
        if feature not in vec_dict:
            continue
        
        value = vec_dict[feature]
        splits = np.array(split_map[feature], dtype=np.float32)
        bucket_idx = int(np.searchsorted(splits, value))
        indices[feature] = bucket_idx
    
    return indices


# def build_bucket_prefilter(
#     bucket_indices: Dict[str, int]
# ) -> Optional[Dict]:
#     """Bucket pre-filtering query 생성 (피처별 그룹핑)"""
#     if not bucket_config or not bucket_indices:
#         return None
    
#     feature_groups = []
    
#     for feature, bucket_idx in bucket_indices.items():
#         # ★ Confidence 계산 (사전계산된 center 활용)
#         if feature in bucket_centers:
#             # 입력 bucket의 center와 실제 값의 거리 (생략 가능)
#             # 여기서는 단순화: 모든 feature에 동일 boost
#             boost = 2.0
#         else:
#             boost = 1.0
        
#         # ★ ±1 Tolerance
#         tolerance_buckets = [
#             max(0, bucket_idx - 1),
#             bucket_idx,
#             bucket_idx + 1
#         ]
        
#         # ★ 피처별 그룹 (양방향 고려)
#         feature_group = {
#             "bool": {
#                 "should": [
#                     {
#                         "bool": {
#                             "must": [
#                                 {"term": {f"{feature}_direction": "lte"}},
#                                 {"terms": {f"{feature}_bucket": tolerance_buckets}}
#                             ]
#                         }
#                     },
#                     {
#                         "bool": {
#                             "must": [
#                                 {"term": {f"{feature}_direction": "gte"}},
#                                 {"terms": {f"{feature}_bucket": tolerance_buckets}}
#                             ]
#                         }
#                     }
#                 ],
#                 "minimum_should_match": 1,
#                 "boost": boost
#             }
#         }
#         feature_groups.append(feature_group)
    
#     # ★ 9개 피처 중 6개 매칭
#     n_features = len(feature_groups)
#     min_match = max(1, int(n_features * 0.67))  # 67%
    
#     return {
#         "bool": {
#             "should": feature_groups,
#             "minimum_should_match": min_match
#         }
#     }

# router.py Line 432-491 수정
def build_bucket_prefilter(bucket_indices: Dict[str, int]) -> Optional[Dict]:
    if not bucket_config or not bucket_indices:
        return None
    
    feature_groups = []
    
    for feature, bucket_idx in bucket_indices.items():
        tolerance_buckets = list(range(max(0, bucket_idx - 2), bucket_idx + 3))
        
        # Direction 유지 (기존 로직)
        feature_group = {
            "bool": {
                "should": [
                    {"bool": {"must": [
                        {"term": {f"{feature}_direction": "lte"}},
                        {"terms": {f"{feature}_bucket": tolerance_buckets}}
                    ]}},
                    {"bool": {"must": [
                        {"term": {f"{feature}_direction": "gte"}},
                        {"terms": {f"{feature}_bucket": tolerance_buckets}}
                    ]}}
                ],
                "minimum_should_match": 1
            }
        }
        feature_groups.append(feature_group)
    
    n_features = len(feature_groups)
    min_match = 3  # 9개 중 3개 (33%) - 고정값
    
    return {"bool": {"should": feature_groups, "minimum_should_match": min_match}}

# ===========================
# Core Search: Top-5 with Fallback
# ===========================

def search_with_fallback(
    index: str,
    test_doc: Dict[str, float],
    experiment_case: str,
    version: str,
) -> Tuple[List[Dict], str]:
    """
    1차: strict percolate → Top-5
    2차: centroid fallback → Top-5
         - percolate 완전 실패(매치 0건) 시 진입
         - test_doc (percolate_doc, PCA 변환 후) 벡터와 각 cluster centroid 간
           cosine distance 최소 기준으로 Top-5 클러스터 결정
         - centroid 파일: shape=(K, D)  D=experiment_case vector_dim
           GCS: {GCS_TREE_BASE}/{EXPERIMENT_CASE}/cluster_centroids.npy
           예) pca_64_k150 → (150, 64),  k150(no-PCA) → (150, 576)
         - 반환 포맷 = strict percolate와 동일 (top5 hit list + match_type="centroid")
    ★ v5: _source: cluster_id, query, persona 만 조회
    """
    SOURCE_FIELDS = ["cluster_id", "query", "persona"]

    # ★ 1. Bucket prefilter 생성 (bucket_config 존재 시)
    bucket_filter = None
    if bucket_config:  # 전역 변수
        # test_doc → numpy array 변환
        vec = np.array(
            [test_doc.get(f"v{i+1}", 0.0) for i in range(len(test_doc))],
            dtype=np.float32
        )
        bucket_indices = calculate_bucket_indices(vec, experiment_case)
        bucket_filter = build_bucket_prefilter(bucket_indices)
        
        if bucket_filter:
            logger.info("bucket_prefilter_generated",  # debug → info로 변경
                    indices=bucket_indices,
                    features=len(bucket_indices),
                    min_match=bucket_filter["bool"]["minimum_should_match"])
    
    # ★ 2. Percolate query 구성
    if bucket_filter:
        query = {
            "bool": {
                "must": [
                    {"percolate": {"field": "query", "document": test_doc}},
                    bucket_filter  # ← term filter로 빠른 pre-filtering
                ]
            }
        }
    else:
        query = {"percolate": {"field": "query", "document": test_doc}}

    # ── 1차: Strict Percolate ────────────────────────────────────
    try:
        result = client.search(
            index=index,
            body={
                "query": query,  # ← bucket prefilter 포함
                "size": 5,
                "_source": SOURCE_FIELDS,
            }
        )
        hits = result["hits"]["hits"]
        if hits:
            match_type = "strict_bucketed" if bucket_filter else "strict"
            match_type_counter.labels(
                experiment_case=experiment_case, version=version, match_type=match_type
            ).inc()
            logger.info(match_type,
                        top_cluster_ids=[h["_source"]["cluster_id"] for h in hits],
                        bucket_filter_used=bool(bucket_filter))
            return hits, match_type
    except Exception as e:
        logger.error("percolate_error", error=str(e), bucket_filter_used=bool(bucket_filter))
        raise HTTPException(status_code=500, detail=f"Percolate search error: {e}")
    
    # try:
    #     result = client.search(
    #         index=index,
    #         body={
    #             "query": {
    #                 "percolate": {
    #                     "field": "query",
    #                     "document": test_doc,
    #                 }
    #             },
    #             "size": 5,
    #             "_source": SOURCE_FIELDS,
    #         }
    #     )
    #     hits = result["hits"]["hits"]
    #     if hits:
    #         match_type_counter.labels(
    #             experiment_case=experiment_case, version=version, match_type="strict"
    #         ).inc()
    #         logger.info("strict_match",
    #                     top_cluster_ids=[h["_source"]["cluster_id"] for h in hits])
    #         return hits, "strict"
    # except Exception as e:
    #     logger.error("percolate_error", error=str(e))
    #     raise HTTPException(status_code=500, detail=f"Percolate search error: {e}")

    # ── 2차: Centroid Fallback ───────────────────────────────────
    # percolate 매치 0건일 때 진입.
    # test_doc (PCA 변환 후 벡터 {v1:..., v2:..., ...}) 과
    # cluster centroid 행렬 (shape=(K, D)) 간 cosine distance 최솟값 기준으로
    # Top-5 클러스터를 결정, percolator 인덱스에서 해당 doc을 조회해 반환.
    #
    # centroid 차원 D = experiment_case 의 vector_dim 과 동일해야 함.
    #   pca_64_k150 → D=64,  k150(emb_vectors) → D=576
    # GCS: {GCS_TREE_BASE}/k{N_CLUSTERS}/cluster_centroids.npy
    match_type_counter.labels(
        experiment_case=experiment_case, version=version, match_type="centroid"
    ).inc()
    t0 = time.perf_counter()

    try:
        if centroid_matrix is None or centroid_cluster_ids is None:
            raise HTTPException(
                status_code=500,
                detail="Centroid matrix not loaded. Check CENTROID_PATH and startup logs."
            )

        # ── test_doc → numpy 벡터 변환 ──────────────────────────
        # test_doc = {"v1": ..., "v2": ..., ...}  (PCA 변환 후, percolate와 동일 공간)
        # centroid_matrix.shape[1] = D (실험 케이스 vector_dim)
        D = centroid_matrix.shape[1]
        test_vec = np.array(
            [test_doc.get(f"v{i+1}", 0.0) for i in range(D)],
            dtype=np.float32,
        )
        if test_vec.shape[0] != D:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Centroid dim mismatch: centroid_dim={D}, "
                    f"test_doc_dim={test_vec.shape[0]}. "
                    f"EXPERIMENT_CASE={experiment_case}의 centroid 파일 차원을 확인하세요."
                )
            )

        # ── cosine distance = 1 - cosine_similarity ────────────
        test_norm = np.linalg.norm(test_vec)
        if test_norm > 0:
            test_vec_normalized = test_vec / test_norm
        else:
            test_vec_normalized = test_vec

        cent_norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
        cent_norms = np.where(cent_norms == 0, 1.0, cent_norms)
        cent_normalized = centroid_matrix / cent_norms

        cosine_similarities = cent_normalized @ test_vec_normalized   # (K,)
        distances           = 1.0 - cosine_similarities               # cosine distance

        # ── Top-5 최근접 centroid 선택 ─────────────────────────
        top5_idx = np.argsort(distances)[:5]

        # centroid row index → ES percolator에서 해당 cluster_id의 _source 조회
        top5_cluster_ids = [centroid_cluster_ids[int(i)] for i in top5_idx]
        top5_distances   = [float(distances[i])          for i in top5_idx]

        # percolator 인덱스에서 top5 cluster_id에 해당하는 doc 조회
        should_clauses = [
            {"term": {"cluster_id": cid}} for cid in top5_cluster_ids
        ]
        centroid_result = client.search(
            index=index,
            body={
                "query": {"bool": {"should": should_clauses, "minimum_should_match": 1}},
                "size":  5,
                "_source": SOURCE_FIELDS,
            }
        )
        centroid_hits = centroid_result["hits"]["hits"]

        # ── 반환 순서를 distance 오름차순으로 정렬 ───────────────
        cid_to_hit   = {h["_source"]["cluster_id"]: h for h in centroid_hits}
        top5_ordered = []
        for cid, dist in zip(top5_cluster_ids, top5_distances):
            if cid in cid_to_hit:
                hit = cid_to_hit[cid]
                # _score를 -distance로 덮어써서 상위 정렬과 호환
                hit["_score"] = float(1.0 - dist)
                top5_ordered.append(hit)

        elapsed = (time.perf_counter() - t0) * 1000
        centroid_fallback_latency.labels(
            experiment_case=experiment_case, version=version
        ).observe(elapsed)

        if not top5_ordered:
            raise HTTPException(status_code=500, detail="Centroid fallback: no cluster docs found")

        logger.info(
            "centroid_match",
            top_cluster_ids=[h["_source"]["cluster_id"] for h in top5_ordered],
            top_distances=top5_distances[:len(top5_ordered)],
            latency_ms=elapsed,
        )
        return top5_ordered, "centroid"

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Centroid fallback error: {e}")

def _augment_with_centroid(
    v1_hits: List[Dict],
    match_type: str,
    test_doc: Dict[str, float],
    index: str,
    experiment_case: str,
    version: str,
) -> Tuple[List[Dict], str]:
    """
    V14 전용: V1 strict match 결과(primary cluster) + centroid 기반 인접 cluster top4 결합.
    
    V1이 1개만 매칭해도 top5를 구성할 수 있도록 centroid_matrix로 보완.
    이미 매칭된 cluster는 중복 제거.
    """
    if not v1_hits or centroid_matrix is None or centroid_cluster_ids is None:
        return v1_hits, match_type

    # V1 매칭된 cluster_id 수집
    matched_cluster_ids = {h["_source"]["cluster_id"] for h in v1_hits}

    # primary cluster의 centroid 기준 인접 cluster 계산
    primary_cid = v1_hits[0]["_source"]["cluster_id"]
    if primary_cid not in centroid_cluster_ids:
        return v1_hits, match_type

    primary_row = centroid_cluster_ids.index(primary_cid)
    primary_vec = centroid_matrix[primary_row]

    # cosine distance 계산
    test_vec = np.array(
        [test_doc.get(f"v{i+1}", 0.0) for i in range(centroid_matrix.shape[1])],
        dtype=np.float32,
    )
    cent_norms  = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
    cent_norms  = np.where(cent_norms == 0, 1.0, cent_norms)
    test_norm   = np.linalg.norm(test_vec)
    test_norm   = test_norm if test_norm > 0 else 1.0
    similarities = (centroid_matrix / cent_norms) @ (test_vec / test_norm)
    distances    = 1.0 - similarities

    # 이미 매칭된 cluster 제외하고 top4 선택
    sorted_idx = np.argsort(distances)
    extra_cluster_ids = []
    for idx in sorted_idx:
        cid = centroid_cluster_ids[int(idx)]
        if cid not in matched_cluster_ids:
            extra_cluster_ids.append(cid)
        if len(extra_cluster_ids) >= (5 - len(v1_hits)):
            break

    if not extra_cluster_ids:
        return v1_hits, match_type

    # percolator에서 extra cluster doc 조회
    should_clauses = [{"term": {"cluster_id": cid}} for cid in extra_cluster_ids]
    extra_result = client.search(
        index=index,
        body={
            "query": {"bool": {"should": should_clauses, "minimum_should_match": 1}},
            "size":  len(extra_cluster_ids),
            "_source": ["cluster_id", "query", "persona"],
        }
    )
    extra_hits = extra_result["hits"]["hits"]
    for hit in extra_hits:
        hit["_score"] = 0.5  # centroid 기반임을 score로 구분

    combined = v1_hits + extra_hits
    return combined[:5], "v14_centroid_augmented"



# ===========================
# Route Core Logic
# ===========================

def _do_route(
    test_case: TestCase,
    selected_case: str,
    selected_version: str,
    target_index: str,
) -> RouteResponse:
    """단일 라우팅 처리 (route / route-batch 공통)"""
    start = time.perf_counter()
    vec_index = get_vec_index()

    # ── 입력 검증 ────────────────────────────────────────────────
    if len(test_case.embedding) != EMB_INPUT_DIM:
        raise HTTPException(
            status_code=422,
            detail=f"embedding dimension must be {EMB_INPUT_DIM}, got {len(test_case.embedding)}"
        )

    # ★ v5: SKIP_ANALYZER=true → percolate 스킵, 전체 cluster_id 직접 반환
    if SKIP_ANALYZER:
        all_cluster_ids = get_all_cluster_ids(target_index)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # cluster_ids를 ClusterCandidate로 변환 (score=1.0 고정, rank 순)
        top_k_clusters = [
            ClusterCandidate(cluster_id=cid, rank=i + 1, score=1.0)
            for i, cid in enumerate(all_cluster_ids[:5])
        ]

        # top_k_clusters = [
        #     ClusterCandidate(cluster_id=cid, rank=i + 1, score=1.0)
        #     for i, cid in enumerate(all_cluster_ids)
        # ]
        
        primary_id = all_cluster_ids[0] if all_cluster_ids else -1

        skip_analyzer_counter.labels(experiment_case=selected_case,cluster_id=str(primary_id)).inc()


        logger.info(
            "routing_skip_analyzer",
            primary_cluster_id=primary_id,
            cluster_count=len(all_cluster_ids),
            vec_index=vec_index,
            latency_ms=elapsed_ms,
        )

        return RouteResponse(
            primary_cluster_id=primary_id,
            top_k_clusters=top_k_clusters,
            vec_index=vec_index,
            is_hot=False,
            cache_ttl=CACHE_TTL_COLD,
            latency_ms=elapsed_ms,
            index_used=target_index,
            experiment_case=selected_case,
            version=selected_version,
            match_type="skip_analyzer",
            match_score=1.0,
            skip_analyzer=True,
            tree_features={},
            persona=None,
            preprocessing_stats=None,
        )
    
    # ── 전처리: embedding → {"v1": ..., "v2": ...} ──────────────
    t_pre = time.perf_counter()
    percolate_doc = preprocess_to_percolate_doc(
        test_case.embedding, selected_case, pca_model
    )
    preprocess_elapsed_ms = (time.perf_counter() - t_pre) * 1000  # ★ 즉시 계산
    preprocessing_latency.labels(experiment_case=selected_case).observe(preprocess_elapsed_ms)
    preprocessing_stats = {
        "input_dim":   len(test_case.embedding),
        "output_dim":  len(percolate_doc),
        "pca_applied": EXPERIMENT_CONFIG.get(selected_case, {}).get("use_pca", False),
    }

    # ── ES Percolate 검색 ────────────────────────────────────────
    percolate_requests.labels(experiment_case=selected_case, version=selected_version).inc()

    # ★ 수정: 중복 호출 제거 + thread 대기 시간 분리
    # t_thread_submit = time.perf_counter()
    t_percolate = time.perf_counter()

    # ★ 수정: V14일 때 centroid로 top5 보완
    top5_hits, match_type = search_with_fallback(
        target_index, percolate_doc, selected_case, selected_version,
    )

    # ★ 수정: V14일 때 centroid로 top5 보완
    if selected_version == "v14":
        logger.info("v14_augment_attempt",
                    centroid_matrix_loaded=centroid_matrix is not None,
                    v1_hits_count=len(top5_hits))
        top5_hits, match_type = _augment_with_centroid(
                top5_hits, match_type, percolate_doc, target_index, selected_case, selected_version
            )
        logger.info("v14_augment_done",
                    match_type=match_type,
                    top5_count=len(top5_hits))
            
    percolate_es_ms = (time.perf_counter() - t_percolate) * 1000

    top_k_clusters = [
        ClusterCandidate(
            cluster_id=h["_source"]["cluster_id"],
            rank=rank,
            score=float(h.get("_score", 0.0)),
        )
        for rank, h in enumerate(top5_hits, start=1)
    ]

    top5_ids       = [c.cluster_id for c in top_k_clusters]
    top5_diversity = len(set(top5_ids)) / len(top5_ids) if top5_ids else 0.0

    # ── Top-1 정보 ────────────────────────────────────────────────
    primary_src   = top5_hits[0]["_source"]
    primary_id    = primary_src["cluster_id"]
    primary_score = float(top5_hits[0].get("_score", 1.0))

    # ★ 추가: tree_features 추출 시간
    t_tree = time.perf_counter()
    tree_features = extract_tree_features(primary_src.get("query", {}))
    tree_features_ms = (time.perf_counter() - t_tree) * 1000

    persona = None
    if PERSONA_ENABLED:
        persona = primary_src.get("persona")
    persona_delivered.labels(has_persona=str(bool(persona)).lower()).inc()

    # ── Hot/Cold 분류 ──────────────────────────────────────────
    t_hot = time.perf_counter()
    update_cluster_frequency(primary_id)
    cluster_hits.labels(
        cluster_id=str(primary_id), experiment_case=selected_case, version=selected_version
    ).inc()
    hot_list = get_hot_clusters()
    is_hot   = primary_id in hot_list
    hot_clusters_count.set(len(hot_list))
    hot_cluster_ms = (time.perf_counter() - t_hot) * 1000

    elapsed_ms = (time.perf_counter() - start) * 1000
    other_ms = elapsed_ms - preprocess_elapsed_ms - percolate_es_ms - tree_features_ms - hot_cluster_ms

    percolate_latency.labels(experiment_case=selected_case, version=selected_version).observe(elapsed_ms)

    logger.info(
        "routing_done",
        primary_cluster_id=primary_id,
        top5=[c.cluster_id for c in top_k_clusters],
        top5_diversity=round(top5_diversity, 4),
        match_type=match_type,
        score=primary_score,
        vec_index=vec_index,
        # ★ 구간별 분해
        latency_ms=elapsed_ms,
        preprocess_ms=preprocess_elapsed_ms,
        percolate_es_ms=percolate_es_ms,
        tree_features_ms=tree_features_ms,
        hot_cluster_ms=hot_cluster_ms,
        other_ms=other_ms,        # thread 대기 + connection pool 등 미분류 잔여
    )

    return RouteResponse(
        primary_cluster_id=primary_id,
        top_k_clusters=top_k_clusters,
        vec_index=vec_index,
        is_hot=is_hot,
        cache_ttl=CACHE_TTL_HOT if is_hot else CACHE_TTL_COLD,
        latency_ms=elapsed_ms,
        index_used=target_index,
        experiment_case=selected_case,
        version=selected_version,
        match_type=match_type,
        match_score=primary_score,
        skip_analyzer=False,
        tree_features=tree_features,
        persona=persona,
        preprocessing_stats=preprocessing_stats,
    )

# ===========================
# Route Endpoints
# ===========================

# route 엔드포인트에서 측정:
@app.post("/route", response_model=RouteResponse)
async def route_to_cluster(
    test_case: TestCase,
    x_router_case:    Optional[str] = Header(None, description="Override experiment_case"),
    x_router_version: Optional[str] = Header(None, description="Override query version (v1-v5)"),
):
    selected_case    = x_router_case    if x_router_case    in AVAILABLE_EXPERIMENTS else EXPERIMENT_CASE
    selected_version = x_router_version if x_router_version in AVAILABLE_VERSIONS    else PERCOLATE_QUERY_VERSION
    # target_index     = f"fraud_ecom_{selected_case}_{selected_version}_tree_rules_percolator"
    index_version    = VERSION_INDEX_ALIAS.get(selected_version, selected_version)  # ★ 추가
    target_index     = f"fraud_ecom_{selected_case}_{index_version}_tree_rules_percolator"

    t_submit = time.perf_counter()
    result = await asyncio.to_thread(
        _do_route, test_case, selected_case, selected_version, target_index
    )
    total_endpoint_ms = (time.perf_counter() - t_submit) * 1000

    # result.latency_ms = _do_route 내부 실행 시간
    thread_wait_ms = total_endpoint_ms - result.latency_ms

    logger.info(
        "route_endpoint_timing",
        total_endpoint_ms=total_endpoint_ms,
        do_route_ms=result.latency_ms,
        thread_wait_ms=thread_wait_ms,  # ← 이게 크면 thread pool 대기
    )
    return result


@app.post("/route-batch", response_model=List[RouteResponse])
async def route_batch(
    test_cases: List[TestCase],
    x_router_case:    Optional[str] = Header(None),
    x_router_version: Optional[str] = Header(None),
):
    if len(test_cases) > 100:
        raise HTTPException(status_code=400, detail="Batch size exceeds limit (max 100)")

    selected_case    = x_router_case    if x_router_case    in AVAILABLE_EXPERIMENTS else EXPERIMENT_CASE
    selected_version = x_router_version if x_router_version in AVAILABLE_VERSIONS    else PERCOLATE_QUERY_VERSION
    # target_index     = f"fraud_ecom_{selected_case}_{selected_version}_tree_rules_percolator"
    index_version    = VERSION_INDEX_ALIAS.get(selected_version, selected_version)  # ★ 추가
    target_index     = f"fraud_ecom_{selected_case}_{index_version}_tree_rules_percolator"

    async def _run(tc: TestCase) -> RouteResponse:
        try:
            return await asyncio.to_thread(
                _do_route, tc, selected_case, selected_version, target_index
            )
        except Exception as e:
            logger.error("batch_item_failed", error=str(e))
            return RouteResponse(
                primary_cluster_id=-1,
                top_k_clusters=[],
                vec_index=get_vec_index(),
                is_hot=False,
                cache_ttl=300,
                latency_ms=0.0,
                index_used=target_index,
                experiment_case=selected_case,
                version=selected_version,
                match_type="error",
                match_score=0.0,
                skip_analyzer=SKIP_ANALYZER,
                tree_features={},
                persona=None,
                preprocessing_stats=None,
            )

    return await asyncio.gather(*[_run(tc) for tc in test_cases])


# ===========================
# Admin / Health Endpoints
# ===========================

@app.get("/hot-clusters")
async def list_hot_clusters():
    hot   = get_hot_clusters()
    total = sum(cluster_frequency.values())
    hot_r = sum(cluster_frequency.get(c, 0) for c in hot)
    return {
        "hot_clusters":         hot,
        "hot_cluster_count":    len(hot),
        "percentile_threshold": HOT_CLUSTER_PERCENTILE,
        "coverage_percent":     round(hot_r / total * 100, 2) if total else 0.0,
    }

@app.get("/health")
async def health():
    return {
        "status":          "healthy",
        "service":         "router",
        "version":         "5.0",
        "experiment_case": EXPERIMENT_CASE,
        "pca_loaded":      pca_model is not None,
        "skip_analyzer":   SKIP_ANALYZER,
        "vec_index":       get_vec_index(),
    }

@app.get("/ready")
async def ready():
    try:
        info = client.cluster.health(timeout="5s")
        return {
            "status":        "ready",
            "elasticsearch": info.get("cluster_name"),
            "configuration": {
                "experiment_case": EXPERIMENT_CASE,
                "vector_dim":      EXPERIMENT_CONFIG.get(EXPERIMENT_CASE, {}).get("vector_dim"),
                "version":         PERCOLATE_QUERY_VERSION,
                "pca_model":       "loaded" if pca_model else "not_required",
                "top_k":           5,
                "persona_enabled": PERSONA_ENABLED,
                "skip_analyzer":   SKIP_ANALYZER,
                "vec_index":       get_vec_index(),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ===========================
# Startup / Shutdown
# ===========================

@app.on_event("startup")
async def startup_event():
    global pca_model, centroid_matrix, centroid_cluster_ids
    global bucket_config, bucket_centers  # ★ NEW

    cfg     = EXPERIMENT_CONFIG.get(EXPERIMENT_CASE, {})
    use_pca = cfg.get("use_pca", False)

    # ── PCA 모델 ─────────────────────────────────────────────────
    if use_pca:
        if os.path.exists(PCA_MODEL_PATH):
            pca_model = joblib.load(PCA_MODEL_PATH)
            logger.info("pca_model_loaded", path=PCA_MODEL_PATH,
                        experiment_case=EXPERIMENT_CASE, vector_dim=cfg.get("vector_dim"))
        else:
            logger.error("pca_model_not_found", path=PCA_MODEL_PATH,
                         experiment_case=EXPERIMENT_CASE)
            raise RuntimeError(
                f"PCA model required for '{EXPERIMENT_CASE}' but not found at {PCA_MODEL_PATH}"
            )
    else:
        logger.info("pca_not_required", experiment_case=EXPERIMENT_CASE,
                    vector_dim=cfg.get("vector_dim"))

    # ── Centroid 행렬 로드 ─────────────────────────────────────────
    # GCS: {GCS_TREE_BASE}/k{N_CLUSTERS}/cluster_centroids.npy
    #   shape=(K, D)  K=클러스터 수, D=experiment_case의 vector_dim (PCA 후)
    #   예) pca_64_k150 → (150, 64),  k150(emb_vectors) → (150, 576)
    # initContainer가 N_CLUSTERS 환경변수로 GCS 경로 구성 후 CENTROID_PATH에 다운로드.
    # centroid row i ↔ cluster_id: npy 생성 시 cluster_id 오름차순 정렬 기준.
    # centroid_cluster_ids: 실제 cluster_id 목록은 percolator 인덱스 agg로 로드.
    if os.path.exists(CENTROID_PATH):
        centroid_matrix = np.load(CENTROID_PATH).astype(np.float32)
        expected_dim = EXPERIMENT_CONFIG.get(EXPERIMENT_CASE, {}).get("vector_dim", EMB_INPUT_DIM)
        if centroid_matrix.shape[1] != expected_dim:
            logger.warning(
                "centroid_dim_mismatch",
                centroid_dim=centroid_matrix.shape[1],
                expected_dim=expected_dim,
                experiment_case=EXPERIMENT_CASE,
                note=(
                    f"centroid shape[1]={centroid_matrix.shape[1]} 이 "
                    f"experiment_case '{EXPERIMENT_CASE}'의 vector_dim={expected_dim}과 다름. "
                    "올바른 k{N}/cluster_centroids.npy 파일인지 확인하세요."
                ),
            )
        logger.info(
            "centroid_loaded",
            path=CENTROID_PATH,
            shape=centroid_matrix.shape,
            experiment_case=EXPERIMENT_CASE,
            expected_dim=expected_dim,
            dim_ok=(centroid_matrix.shape[1] == expected_dim),
        )
        # bucketization 점검
        try:            
            _perc_idx = f"fraud_ecom_{EXPERIMENT_CASE}_{PERCOLATE_QUERY_VERSION}_tree_rules_percolator"
            mapping = client.indices.get_mapping(index=_perc_idx)
            _meta = mapping[_perc_idx]["mappings"].get("_meta", {})
            # ★ 명시적 disable 체크
            if ENABLE_BUCKETIZATION == "false":
                logger.info("bucket_config_disabled", reason="env ENABLE_BUCKETIZATION=false")       
            else:
                if ENABLE_BUCKETIZATION == "true":
                    if "bucket_config" in _meta:
                        bucket_config = _meta["bucket_config"]
                        # ★ Bucket center 사전계산 (NumPy 벡터화)
                        bucket_centers = {}
                        for feature, splits in bucket_config["split_points"].items():
                            splits_arr = np.array(splits, dtype=np.float32)
                            # 각 bucket의 중심값 = (left + right) / 2
                            centers = (splits_arr[:-1] + splits_arr[1:]) / 2
                            # 첫/마지막 bucket은 경계값 사용
                            centers = np.concatenate([
                                [splits_arr[0] - 1.0],  # bucket_0 중심
                                centers,
                                [splits_arr[-1] + 1.0]  # bucket_N 중심
                            ])
                            bucket_centers[feature] = centers
                        logger.info(
                            "bucket_config_loaded",
                            version=bucket_config["version"],
                            depth_pct=bucket_config["depth_pct"],
                            features=len(bucket_config["split_points"]),
                            )     
                    else:
                        raise RuntimeError(
                            f"ENABLE_BUCKETIZATION=true but bucket_config not found in _meta"
                        )
                    logger.info("bucket_config_not_found", note="Bucketization disabled")
        except Exception as e:
            if ENABLE_BUCKETIZATION == "true":
                raise
            logger.warning("bucket_config_load_failed", error=str(e))
        # percolator 인덱스에서 cluster_id 목록 로드 (centroid row i ↔ cluster_id 매핑)
        # npy 생성 시 cluster_id 오름차순 정렬 기준으로 row가 배치되어 있어야 함.
        try:
            _perc_idx = f"fraud_ecom_{EXPERIMENT_CASE}_{PERCOLATE_QUERY_VERSION}_tree_rules_percolator"
            _result   = client.search(
                index=_perc_idx,
                body={
                    "query":   {"match_all": {}},
                    "size":    0,
                    "aggs":    {"all_cluster_ids": {"terms": {"field": "cluster_id", "size": 10000}}},
                    "_source": False,
                }
            )
            _buckets = _result["aggregations"]["all_cluster_ids"]["buckets"]
            # centroid row 순서 = cluster_id 오름차순 정렬 (npy 생성 규칙과 동일해야 함)
            centroid_cluster_ids = sorted([int(b["key"]) for b in _buckets])
            logger.info(
                "centroid_cluster_ids_loaded",
                n_clusters=len(centroid_cluster_ids),
                centroid_rows=centroid_matrix.shape[0],
                match=len(centroid_cluster_ids) == centroid_matrix.shape[0],
            )
            if len(centroid_cluster_ids) != centroid_matrix.shape[0]:
                logger.warning(
                    "centroid_row_cluster_id_mismatch",
                    centroid_rows=centroid_matrix.shape[0],
                    n_cluster_ids=len(centroid_cluster_ids),
                    note="centroid_cluster_ids가 centroid_matrix 행 수와 다름 — 인덱스 불일치 가능",
                )
        except Exception as e:
            logger.warning(
                "centroid_cluster_ids_fallback",
                error=str(e),
                note="percolator에서 cluster_id 로드 실패 → 0-indexed 폴백 사용",
            )
            centroid_cluster_ids = list(range(centroid_matrix.shape[0]))
    else:
        logger.warning(
            "centroid_not_found",
            path=CENTROID_PATH,
            note="centroid fallback 불가 — percolate 실패 시 500 에러 반환됨",
        )

    # ★ v5: SKIP_ANALYZER=true 시 startup에서 cluster_id 목록 미리 로드
    if SKIP_ANALYZER:
        _idx = f"fraud_ecom_{EXPERIMENT_CASE}_{PERCOLATE_QUERY_VERSION}_tree_rules_percolator"
        ids  = get_all_cluster_ids(_idx)
        logger.info("skip_analyzer_mode", cluster_count=len(ids), vec_index=get_vec_index())

    logger.info("router_started", version="5.0",
                experiment_case=EXPERIMENT_CASE,
                query_version=PERCOLATE_QUERY_VERSION,
                index=INDEX_PERCOLATOR,
                skip_analyzer=SKIP_ANALYZER,
                vec_index=get_vec_index())

@app.on_event("shutdown")
async def shutdown_event():
    client.close()
    logger.info("router_stopped")

@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    logger.error("validation_error", errors=exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

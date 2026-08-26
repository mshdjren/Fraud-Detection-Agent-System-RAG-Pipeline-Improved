"""
Gaussian Augmented Data Ingestion - V2.2
===================================

[V2.2 변경사항]
  ★ GCS 파일 명명 규칙 수정 (실제 경로 기준)
    - GCS_GAUSSIAN_AUG_BASE: gaussian_aug_v5 → gaussian_aug
    - npy 파일명:
        non-chunk: {EMB_FILENAME_MIDDLE}_percentage_{pct}_mult{mult}.npy
        chunk:     {EMB_FILENAME_MIDDLE}_percentage_{pct}_mult{mult}_chunk_{N}.npy
    - parquet 파일명:
        non-chunk: raw_plus_cluster_tree_percentage_{pct}_mult{mult}.parquet
        chunk:     raw_plus_cluster_tree_percentage_{pct}_mult{mult}_chunk_{N}.parquet

[V2.1 변경사항]
  ★ persona 관련 코드 완전 제거
      - aug 인덱스는 raw vec 데이터 전용, persona는 percolator 인덱스 전용
      - persona 계산/저장은 ingest_tree_pipeline.py에서만 수행

[V2.0 변경사항]
  ① original_index → 매핑에 명시적 추가 (index=False, doc_values=False)
      - aug parquet의 original_index 값을 _source에 저장
      - 원본 row와의 연결 추적용 (GT 매핑, 디버깅)
  ② ES doc _id: es_doc_id 사용 (유지 — aug 고유 ID)
  ③ tree_id: 매핑 및 _source에서 완전 제거

[설계 원칙 - V1 유지]
  ★ 인덱스명: fraud_ecom_aug_{mult}x_{pct}pct_{vec_type}_vec
  ★ row_pos 기반 embedding 참조 (parquet 행 위치)
  ★ vec: index=False (Exact KNN / script_score)
  ★ USE_CHUNK 모드: 대용량(20x/30x) chunk 순차 처리

Usage:
    GAUSSIAN_AUG_MULTIPLIERS=2,5,10 CORESET_PERCENTAGES=100,10,1 \\
        python ingest_gaussian_aug.py

    VEC_ELEMENT_TYPE=byte GAUSSIAN_AUG_MULTIPLIERS=5,10 \\
        python ingest_gaussian_aug.py

    USE_CHUNK=true GAUSSIAN_AUG_MULTIPLIERS=20,30 CORESET_PERCENTAGES=100 \\
        N_CHUNKS=5 python ingest_gaussian_aug.py
"""

import os
import time
import json
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
# 50
from elasticsearch.helpers import parallel_bulk          # ★ streaming_bulk → parallel_bulk
# ===========================
# Configuration
# ===========================

ES_URL   = os.getenv("ES_URL")
PASSWORD = os.getenv("PASSWORD")

GAUSSIAN_AUG_DATA_DIR = os.getenv("GAUSSIAN_AUG_DATA_DIR", "/data/gaussian_aug")
GCS_GAUSSIAN_AUG_BASE = os.getenv("GCS_GAUSSIAN_AUG_BASE",
                                   "gs://fraud_ecommm/tree-search/gaussian_aug")

GAUSSIAN_AUG_MULTIPLIERS = [
    int(m.strip())
    for m in os.getenv("GAUSSIAN_AUG_MULTIPLIERS", "2,5,10,20,30").split(",")
    if m.strip()
]
CORESET_PERCENTAGES = [
    int(p.strip())
    for p in os.getenv("CORESET_PERCENTAGES", "100,10,1").split(",")
    if p.strip()
]

# 변경 후: pct==100이고 mult>=20일 때만 chunk, 나머지는 single
CHUNK_THRESHOLD_MULT = int(os.getenv("CHUNK_THRESHOLD_MULT", "20"))  # Configuration 섹션에 추가
CHUNK_ONLY_PCT       = int(os.getenv("CHUNK_ONLY_PCT", "100"))       # chunk 대상 pct


VEC_ELEMENT_TYPE = "float"

VEC_DIMS = int(os.getenv("VEC_DIMS", "576"))

USE_CHUNK = os.getenv("USE_CHUNK", "false").strip().lower() == "true"
N_CHUNKS  = int(os.getenv("N_CHUNKS", "5"))

EMB_FILENAME_MIDDLE = os.getenv(
    "EMB_FILENAME_MIDDLE",
    "anollm_lr5e-05_standard_smolLM_train_embeddings_percentage"
)

# 93-95행 근처 추가
BULK_THREAD_COUNT = int(os.getenv("BULK_THREAD_COUNT", "4"))   # ★ [ADD]
CHUNK_SIZE      = int(os.getenv("BULK_CHUNK_SIZE",      "1000"))
MAX_CHUNK_BYTES = int(os.getenv("BULK_MAX_CHUNK_BYTES",  str(20 * 1024 * 1024)))
REQUEST_TIMEOUT = int(os.getenv("ES_REQUEST_TIMEOUT",    "120"))

# After — ★ [FIX] router.py get_vec_index()의 VEC_TYPE env와 1:1 대응.
#   미설정 시 기존 동작과 완전히 동일(float32/int8) — 하위호환 100% 유지.
#   2-stage(bbq) job에서만 VEC_TYPE=bbq 로 override하면 router도 동일 env로
#   같은 인덱스명을 찾으므로 router.py 코드 변경이 필요 없다.
# After — ★ [FIX] VEC_QUANT을 단일 소스로. 명시 override 안 하면 자동 파생.
VEC_QUANT_LABELS = {"f32": "float32", "int8_hnsw": "int8", "bbq_hnsw": "bbq", "int4_hnsw": "int4"}
VEC_QUANT        = os.getenv("VEC_QUANT", "f32").strip().lower()
VEC_TYPE_LABEL = os.getenv("VEC_TYPE", VEC_QUANT_LABELS.get(VEC_QUANT, VEC_QUANT))
# VEC_TYPE_LABEL = os.getenv("VEC_TYPE", "int8" if VEC_ELEMENT_TYPE == "byte" else "float32")

# ★ V2.1: persona 관련 env/코드 완전 제거
#   aug 인덱스는 raw vec 데이터만 저장 (persona는 percolator 인덱스 전용).
#   persona 계산/저장/업로드는 ingest_tree_pipeline.py에서만 수행.

# ★ aug vec 인덱스 shard 분산
INGEST_SEARCH_MODE = os.getenv("INGEST_SEARCH_MODE", "knn").strip().lower()

AUG_SHARD_DISTRIBUTION = os.getenv("AUG_SHARD_DISTRIBUTION", "false").strip().lower() == "true"
ATTR_DOCS  = os.getenv("ATTR_DOCS", "docs")

N_SHARDS_VEC  = int(os.getenv("N_SHARDS_VEC", "3"))
N_REPLICAS  = int(os.getenv("N_REPLICAS", "0"))

# ★ [ADD] 1stage(기존 검증된 경로) / 2stage(BBQ+rescore, 신규) 명시 분리
AUG_RETRIEVAL_MODE = os.getenv("AUG_RETRIEVAL_MODE", "1stage").strip().lower()  # "1stage" | "2stage"
HNSW_M     = int(os.getenv("HNSW_M", "16"))
HNSW_EF_C  = int(os.getenv("HNSW_EF_CONSTRUCTION", "100"))
_QUANT_INDEX_TYPE = {"f32": "hnsw", "int8_hnsw": "int8_hnsw",
                      "int4_hnsw": "int4_hnsw", "bbq_hnsw": "bbq_hnsw"}

# RESUME_INGEST = os.getenv("RESUME_INGEST", "false").strip().lower() == "true"   # ★ [ADD]
RESUME_INGEST = "true"   # ★ [ADD]

print(f"\n{'='*80}")
print(f"GAUSSIAN AUG INGESTION V2.1")
print(f"{'='*80}")
print(f"Multipliers     : {GAUSSIAN_AUG_MULTIPLIERS}")
print(f"Coreset pcts    : {CORESET_PERCENTAGES}")
print(f"Vec type        : {VEC_ELEMENT_TYPE} ({VEC_TYPE_LABEL})")
print(f"Vec dims        : {VEC_DIMS}")
print(f"USE_CHUNK       : {USE_CHUNK}  (N_CHUNKS={N_CHUNKS})")
print(f"Data dir        : {GAUSSIAN_AUG_DATA_DIR}")
print()

# ===========================
# Elasticsearch Client
# ===========================

client = Elasticsearch(
    [ES_URL],
    verify_certs=False,
    ssl_show_warn=False,
    basic_auth=("elastic", PASSWORD),
    request_timeout=REQUEST_TIMEOUT,
)
print(f"[es] Connected: {client.info()['cluster_name']}")

# ===========================
# Path Helpers
# ===========================

def local_parquet_path(mult: int, pct: int, chunk: int = None) -> str:
    # GCS 파일명과 동일하게 로컬 저장
    # non-chunk: raw_plus_cluster_tree_percentage_{pct}_mult{mult}.parquet
    # chunk:     raw_plus_cluster_tree_percentage_{pct}_mult{mult}_chunk_{N}.parquet
    if chunk is None:
        fname = f"raw_plus_cluster_tree_percentage_{pct}_mult{mult}.parquet"
    else:
        fname = f"raw_plus_cluster_tree_percentage_{pct}_mult{mult}_chunk_{chunk}.parquet"
    return os.path.join(GAUSSIAN_AUG_DATA_DIR, fname)


def local_emb_path(mult: int, pct: int, chunk: int = None) -> str:
    # GCS 파일명과 동일하게 로컬 저장
    # non-chunk: {EMB_FILENAME_MIDDLE}_percentage_{pct}_mult{mult}.npy
    # chunk:     {EMB_FILENAME_MIDDLE}_percentage_{pct}_mult{mult}_chunk_{N}.npy
    if chunk is None:
        fname = f"{EMB_FILENAME_MIDDLE}_percentage_{pct}_mult{mult}.npy"
    else:
        fname = f"{EMB_FILENAME_MIDDLE}_percentage_{pct}_mult{mult}_chunk_{chunk}.npy"
    return os.path.join(GAUSSIAN_AUG_DATA_DIR, fname)


def index_name(mult: int, pct: int) -> str:
    return f"fraud_ecom_aug_{mult}x_{pct}pct_{VEC_TYPE_LABEL}_vec"


def gcs_download(gcs_path: str, local_path: str):
    """gsutil 대신 google-cloud-storage Python 클라이언트 사용."""
    from google.cloud import storage

    if os.path.exists(local_path):
        print(f"  [skip] already exists: {local_path}")
        return

    print(f"  [gcs] download: {gcs_path} → {local_path}")

    # gs://bucket/path/to/file 파싱
    assert gcs_path.startswith("gs://"), f"Invalid GCS path: {gcs_path}"
    without_prefix = gcs_path[len("gs://"):]
    bucket_name, blob_name = without_prefix.split("/", 1)

    client  = storage.Client()
    bucket  = client.bucket(bucket_name)
    blob    = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    print(f"  [gcs] ✅  {local_path}  ({os.path.getsize(local_path):,} bytes)")

def gcs_upload(local_path: str, gcs_path: str):                      # ★ [ADD]
    from google.cloud import storage
    assert gcs_path.startswith("gs://")
    bucket_name, blob_name = gcs_path[len("gs://"):].split("/", 1)
    storage.Client().bucket(bucket_name).blob(blob_name).upload_from_filename(local_path)

def _checkpoint_gcs_path(idx: str) -> str:                            # ★ [ADD]
    return f"{GCS_GAUSSIAN_AUG_BASE}/_checkpoints/{idx}.json"

def _load_checkpoint(idx: str) -> dict:                               # ★ [ADD]
    """RESUME_INGEST=false면 항상 빈 dict → 처음부터 진행."""
    if not RESUME_INGEST:
        return {}
    from google.cloud import storage
    gcs_path = _checkpoint_gcs_path(idx)
    bucket_name, blob_name = gcs_path[len("gs://"):].split("/", 1)
    blob = storage.Client().bucket(bucket_name).blob(blob_name)
    return json.loads(blob.download_as_text()) if blob.exists() else {}

def _save_checkpoint(idx: str, last_completed_chunk: int, cumulative_offset: int):  # ★ [ADD]
    tmp = f"/tmp/_ckpt_{idx}.json"
    with open(tmp, "w") as f:
        json.dump({"last_completed_chunk": last_completed_chunk,
                    "cumulative_offset": cumulative_offset}, f)
    gcs_upload(tmp, _checkpoint_gcs_path(idx))


def gcs_parquet_path(mult: int, pct: int, chunk: int = None) -> str:
    # GCS 실제 경로: gs://fraudecom/tree-search/gaussian_aug/
    # non-chunk: raw_plus_cluster_tree_percentage_{pct}_mult{mult}.parquet
    # chunk:     raw_plus_cluster_tree_percentage_{pct}_mult{mult}_chunk_{N}.parquet
    if chunk is None:
        fname = f"raw_plus_cluster_tree_percentage_{pct}_mult{mult}.parquet"
    else:
        fname = f"raw_plus_cluster_tree_percentage_{pct}_mult{mult}_chunk_{chunk}.parquet"
    return f"{GCS_GAUSSIAN_AUG_BASE}/{fname}"


def gcs_emb_path(mult: int, pct: int, chunk: int = None) -> str:
    # GCS 실제 경로: gs://fraudecom/tree-search/gaussian_aug/
    # non-chunk: {EMB_FILENAME_MIDDLE}_percentage_{pct}_mult{mult}.npy
    # chunk:     {EMB_FILENAME_MIDDLE}_percentage_{pct}_mult{mult}_chunk_{N}.npy
    if chunk is None:
        fname = f"{EMB_FILENAME_MIDDLE}_percentage_{pct}_mult{mult}.npy"
    else:
        fname = f"{EMB_FILENAME_MIDDLE}_percentage_{pct}_mult{mult}_chunk_{chunk}.npy"
    return f"{GCS_GAUSSIAN_AUG_BASE}/{fname}"

# ===========================
# Index Mapping
# ===========================

# def gen_mapping(df_sample: pd.DataFrame) -> dict:
def gen_mapping(df_sample: pd.DataFrame, box_type: str = None) -> dict: 
    props = {}
    exclude_cols = {"tree_id"}
    no_search_cols = {"label", "original_index", "es_doc_id"}

    RETRIEVAL_FILTERABLE_COLS = {"sex", "source", "browser", "IP_country",
                                "month_purchase", "weekday_purchase"}
    
    for col in df_sample.columns:
        # ★ 컬럼명의 공백을 언더바(_)로 변환하여 원본 파이프라인과 일치시킴
        col_norm = col.replace(" ", "_")
        
        if col in exclude_cols or col_norm in exclude_cols:
            continue
            
        if col_norm in RETRIEVAL_FILTERABLE_COLS:
            props[col_norm] = {"type": "keyword"} # 정상적인 keyword 매핑
            continue

        if col_norm == "cluster_id":
            props[col_norm] = {"type": "integer"}
        elif col_norm == "leaf_id":
            props[col_norm] = {"type": "integer", "doc_values": False}
        elif col_norm in no_search_cols:
            props[col_norm] = {"type": "integer", "index": False, "doc_values": False}
        elif np.issubdtype(df_sample[col].dtype, np.integer):
            props[col_norm] = {"type": "integer", "index": False, "doc_values": False}
        elif np.issubdtype(df_sample[col].dtype, np.floating):
            props[col_norm] = {"type": "float", "index": False, "doc_values": False}
        else:
            props[col_norm] = {"type": "keyword", "index": False, "doc_values": False}

    if INGEST_SEARCH_MODE=='knn':
        props["vec"] = {
        "type": "dense_vector", "dims": 576,
        "index": False, 
        # "similarity": "l2_norm"  # ★ 핵심: L2 거리 기반 인덱싱으로 변경
        "element_type": "float",
        }
    else:
        props["vec"] = {
        "type": "dense_vector", "dims": 576,
        "index": True, 
        "similarity": "l2_norm",  # ★ 핵심: L2 거리 기반 인덱싱으로 변경
        "element_type": "float"
        }

    mapping = {"mappings": {"_source": {"excludes": ["vec"]}, "properties": props}}

    settings = {
        "number_of_shards": N_SHARDS_VEC if AUG_SHARD_DISTRIBUTION else 1,
        "number_of_replicas": N_REPLICAS,
        "index.store.preload": ["vec", "cluster_id"],       # ★ [ADD]
    }
    if box_type:
        settings["index.routing.allocation.include.box_type"] = box_type

    mapping["settings"] = settings

    # props["vec"] = {
    #         "type": "dense_vector", 
    #         "dims": VEC_DIMS,
    #         "index": True, 
    #         "similarity": "l2_norm",  # ★ 'cosine'에서 'l2_norm'으로 변경 필요
    #         "element_type": VEC_ELEMENT_TYPE,
    #     }
    return mapping

# ★ [ADD] 2-stage 전용 — index_options로 BBQ 명시, rescore_vector 전제 조건
#   (element_type은 반드시 float 유지 — byte로 사전 양자화하면 rescore용
#    원본 f32가 사라져서 retriever의 rescore_vector가 작동 불가)
def gen_mapping_2stage(df_sample: pd.DataFrame, box_type: str = None) -> dict:
    mapping = gen_mapping(df_sample, box_type)          # 1stage와 동일한 비-vec 필드 재사용
    mapping["mappings"]["properties"]["vec"] = {
        "type": "dense_vector", "dims": VEC_DIMS,
        "index": True,
        "similarity": "l2_norm",
        "element_type": "float",
        "index_options": {
            "type": _QUANT_INDEX_TYPE.get(VEC_QUANT, "bbq_hnsw"),
            "m": HNSW_M,
            "ef_construction": HNSW_EF_C,
        },
    }
    mapping["mappings"]["_meta"] = {"vec_quant": VEC_QUANT, "retrieval_mode": "2stage"}
    return mapping

def _existing_ids(idx: str) -> set:
    """★ [ADD] 재개용 — 이미 인덱싱된 _id 전체를 scroll로 수집.
    doc_values/source 없이 _id만 가져오므로 2~12M 규모에서도 몇 분 내로 끝난다."""
    ids = set()
    if not client.indices.exists(index=idx):
        return ids
    resp = client.search(index=idx, scroll="2m", size=5000,
                          body={"_source": False, "query": {"match_all": {}}})
    scroll_id = resp["_scroll_id"]
    hits = resp["hits"]["hits"]
    while hits:
        ids.update(h["_id"] for h in hits)
        resp = client.scroll(scroll_id=scroll_id, scroll="2m")
        scroll_id = resp["_scroll_id"]
        hits = resp["hits"]["hits"]
    client.clear_scroll(scroll_id=scroll_id)
    return ids
# ===========================
# int8 양자화
# ===========================

# def to_int8(vec: np.ndarray) -> list:
#     norm = np.linalg.norm(vec)
#     if norm > 0:
#         vec = vec / norm
#     return np.clip(np.round(vec * 127), -127, 127).astype(int).tolist()

# ===========================
# Bulk Action Generator
# ===========================

# def gen_actions(df: pd.DataFrame, emb: np.ndarray, idx_name: str):
# After
def gen_actions(df: pd.DataFrame, emb: np.ndarray, idx_name: str, id_offset: int = 0,
                 skip_ids: set = None):                              # ★ [ADD]
    """
    ★ V2.0 변경:
      - tree_id: _source에서 완전 제거
      - original_index: _source에 유지 (int 변환)
      - ES _id: es_doc_id (유지)
      - 임베딩: row_pos 기반 (변경 없음)
    """
    # use_int8  = (VEC_ELEMENT_TYPE == "byte")
    skip_cols = {"tree_id"}   # ★ tree_id 제거
    col_names = list(df.columns)

    for row_pos, row in enumerate(df.itertuples(index=False, name=None)):
        rec = dict(zip(col_names, row))

        # ★ tree_id 제거
        for col in skip_cols:
            rec.pop(col, None)  

        # ES _id: es_doc_id (aug 고유 ID)
        # doc_id = str(int(rec["es_doc_id"]))
        # 361행(doc_id 계산) 바로 다음에 추가
        doc_id = str(int(rec["es_doc_id"]) + id_offset)
        if skip_ids and doc_id in skip_ids:                          # ★ [ADD]
            continue                                                  # 이미 인덱싱됨 — 재전송 생략
        cluster_id = int(rec.get('cluster_id', 0))

        # 정수형 보장
        for col in ("cluster_id", "leaf_id", "label", "original_index", "es_doc_id"):
            if col in rec and rec[col] is not None:
                try:
                    rec[col] = int(rec[col])
                except (ValueError, TypeError):
                    pass

        # row_pos 기반 embedding 참조
        if row_pos >= len(emb):
            print(f"[warn] row_pos={row_pos} >= emb size={len(emb)}, skip")
            continue

        raw_vec = emb[row_pos].astype(np.float32, copy=False)
        rec["vec"] = raw_vec.tolist()

        yield {
            "_op_type": "index",
            "_index":   idx_name,
            "_id":      doc_id,
            "_routing": str(cluster_id),  # 같은 cluster_id는 같은 샤드로 유도
            "_source":  rec,
        }

# ===========================
# Bulk Ingestion
# ===========================

def bulk_ingest(actions, description: str) -> tuple:
    t0 = time.perf_counter()
    ok_count, fail_count, first_errors = 0, 0, []
    # for ok, item in streaming_bulk(
    #     client, actions,
    #     chunk_size=CHUNK_SIZE, max_chunk_bytes=MAX_CHUNK_BYTES,
    #     raise_on_error=False, raise_on_exception=False,
    # ):
    for ok, item in parallel_bulk(                         # ★ [FIX] streaming_bulk → parallel_bulk
        client, actions,
        chunk_size=CHUNK_SIZE, max_chunk_bytes=MAX_CHUNK_BYTES,
        thread_count=BULK_THREAD_COUNT,                     # ★ [ADD]
        raise_on_error=False, raise_on_exception=False,
    ):
        if ok:
            ok_count += 1
            if ok_count % 10000 == 0:
                print(f"    Progress: {ok_count:,} indexed...", flush=True)
        else:
            fail_count += 1
            if len(first_errors) < 3:
                first_errors.append(item)
    elapsed = time.perf_counter() - t0
    print(f"  {description}: ok={ok_count:,}, fail={fail_count}, time={elapsed:.2f}s")
    if first_errors:
        for e in first_errors[:2]:
            print(f"    Error sample: {e}")
    return ok_count, fail_count

# bulk_ingest() 뒤에 추가 (405행 이후)
def _pre_ingest_tune(idx: str):                             # ★ [ADD]
    client.indices.put_settings(index=idx, body={"index": {
        "refresh_interval": "-1", "translog.durability": "async",
        "translog.sync_interval": "30s"}})

def _post_ingest_tune(idx: str):                            # ★ [ADD]
    client.indices.put_settings(index=idx, body={"index": {"refresh_interval": "1s"}})
    client.indices.forcemerge(index=idx, max_num_segments=1, request_timeout=1800)

def log_index_stats(idx_name: str):
    try:
        client.indices.refresh(index=idx_name)
        stats = client.cat.indices(index=idx_name, format="json")
        if stats:
            docs = stats[0].get("docs.count", "?")
            size = stats[0].get("store.size", "?")
            print(f"  [stats] docs={docs}, size={size}")
    except Exception as e:
        print(f"  [stats] error: {e}")

# ===========================
# Single File Processing
# ===========================

def process_single(mult: int, pct: int, routing_config: dict):
    pq_path  = local_parquet_path(mult, pct)
    emb_path = local_emb_path(mult, pct)

    if not os.path.exists(pq_path):
        raise FileNotFoundError(f"Parquet not found: {pq_path}")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embedding not found: {emb_path}")

    print(f"\n[parquet] Loading: {pq_path}")
    df = pd.read_parquet(pq_path)
    print(f"  Shape: {df.shape}")

    print(f"[emb] Loading: {emb_path}")
    emb = np.load(emb_path, mmap_mode='r')
    print(f"  Shape: {emb.shape}")

    if len(df) != len(emb):
        raise ValueError(f"Row count mismatch: parquet={len(df):,}, emb={len(emb):,}")

    idx = index_name(mult, pct)
    mapping = (gen_mapping_2stage(df, **routing_config) if AUG_RETRIEVAL_MODE == "2stage"
               else gen_mapping(df, **routing_config))

    if RESUME_INGEST and client.indices.exists(index=idx):
        print(f"\n[resume] index {idx} 이미 존재 — 삭제하지 않고 기존 _id 수집 중...")
        existing_ids = _existing_ids(idx)
        print(f"[resume] 기존 {len(existing_ids):,}건 확인, 나머지만 bulk")
    else:
        print(f"\n[es] Creating index: {idx}")
        client.options(ignore_status=[400, 404]).indices.delete(index=idx)
        client.indices.create(index=idx, body=mapping)
        existing_ids = set()
    _pre_ingest_tune(idx)
    print(f"  ✅ vec_type={VEC_ELEMENT_TYPE}, dims={VEC_DIMS}, expected_docs={len(df):,}")

    bulk_ingest(gen_actions(df, emb, idx, skip_ids=existing_ids),
                f"{mult}x_{pct}pct_{VEC_TYPE_LABEL}")
    _post_ingest_tune(idx)
    log_index_stats(idx)
    del emb, df

# ===========================
# Chunk Processing
# ===========================

def process_chunks(mult: int, pct: int, routing_config: dict):
    idx = index_name(mult, pct)
    ckpt = _load_checkpoint(idx)
    start_chunk = ckpt.get("last_completed_chunk", -1) + 1
    id_offset   = ckpt.get("cumulative_offset", 0)

    print(f"\n[chunk] {idx}  n_chunks={N_CHUNKS}  mode=streaming(download→bulk→delete)")
    if start_chunk > 0:
        print(f"[resume] checkpoint 발견 — chunk {start_chunk}부터 재개 (id_offset={id_offset:,})")

    pq0 = local_parquet_path(mult, pct, chunk=0)
    gcs_download(gcs_parquet_path(mult, pct, chunk=0), pq0)
    # After — ★ [FIX] start_chunk==0(처음 실행)일 때만 매핑생성+삭제/재생성.
    #   재개 시엔 기존 인덱스를 그대로 씀 — 여기서 delete하면 지금까지 들어간
    #   200만 건이 다시 날아간다.
    if start_chunk == 0:
        gcs_download(gcs_parquet_path(mult, pct, chunk=0), pq0)
        df0 = pd.read_parquet(pq0)
        mapping = (gen_mapping_2stage(df0, **routing_config) if AUG_RETRIEVAL_MODE == "2stage"
                   else gen_mapping(df0, **routing_config))
        del df0
        os.remove(pq0)
        client.options(ignore_status=[400, 404]).indices.delete(index=idx)
        client.indices.create(index=idx, body=mapping)
        print(f"  ✅ Created — vec_type={VEC_ELEMENT_TYPE}, dims={VEC_DIMS}")
    else:
        assert client.indices.exists(index=idx), \
            f"체크포인트는 있는데 인덱스 {idx}가 없음 — GCS 체크포인트와 ES 상태 불일치, 확인 필요"
        print(f"[resume] 기존 인덱스 유지 — 삭제하지 않음")
    _pre_ingest_tune(idx)

    # ── 2. 청크별 스트리밍: 다운로드 → bulk → 즉시 삭제 ──────────
# After — ★ [FIX] 이미 끝난 청크는 건너뜀
    for chunk_i in range(start_chunk, N_CHUNKS):
        pq_path  = local_parquet_path(mult, pct, chunk=chunk_i)
        emb_path = local_emb_path(mult, pct, chunk=chunk_i)

        # ★ 항상 다운로드 (이전 청크 처리 후 삭제됐으므로 파일 없음)
        print(f"\n  [chunk {chunk_i}/{N_CHUNKS-1}] downloading...")
        gcs_download(gcs_parquet_path(mult, pct, chunk=chunk_i), pq_path)
        gcs_download(gcs_emb_path(mult, pct, chunk=chunk_i), emb_path)

        print(f"  [chunk {chunk_i}/{N_CHUNKS-1}] loading...")
        df  = pd.read_parquet(pq_path)
        emb = np.load(emb_path)

        print(f"    parquet: {df.shape}, emb: {emb.shape}")
        if len(df) != len(emb):
            raise ValueError(
                f"Chunk {chunk_i} row mismatch: parquet={len(df):,}, emb={len(emb):,}"
            )

        # bulk_ingest(gen_actions(df, emb, idx),
        #             f"chunk {chunk_i}/{N_CHUNKS-1} ({mult}x_{pct}pct)")
        # 청크 다운로드 직후, bulk_ingest() 호출 전에 추가해볼 수 있는 안전장치
        dup = df["es_doc_id"].duplicated().sum()
        if dup:
            print(f"[warn] chunk {chunk_i}: es_doc_id 중복 {dup}건 — parallel_bulk 결과가 비결정적일 수 있음")
            
        # After — ★ [FIX] 재개 시작 청크(크래시로 중간에 끊겼을 수 있는 그 청크)만
        #   ES에 이미 있는 _id를 조회해서 스킵. 그 이후 청크는 처음 처리하는 것이므로 불필요.
        skip_ids = _existing_ids(idx) if (chunk_i == start_chunk and start_chunk > 0) else None
        bulk_ingest(
            gen_actions(df, emb, idx, id_offset=id_offset, skip_ids=skip_ids),
            f"chunk {chunk_i}/{N_CHUNKS-1} ({mult}x_{pct}pct)"
        )
        id_offset += len(df)
        del emb, df

        os.remove(emb_path)
        os.remove(pq_path)
        _save_checkpoint(idx, chunk_i, id_offset)          # ★ [ADD] 청크 완료마다 GCS에 기록
        free_gb = os.statvfs(GAUSSIAN_AUG_DATA_DIR).f_bavail \
                  * os.statvfs(GAUSSIAN_AUG_DATA_DIR).f_frsize / 1024**3
        print(f"  🗑  deleted chunk {chunk_i}  |  disk free: {free_gb:.1f} GB")

    _post_ingest_tune(idx)
    log_index_stats(idx)

# ===========================
# Main
# ===========================

def main():
    start = time.perf_counter()

    combos = [(mult, pct)
              for mult in GAUSSIAN_AUG_MULTIPLIERS
              for pct  in CORESET_PERCENTAGES]

    print(f"\nTotal combinations: {len(combos)}")
    for mult, pct in combos:
        print(f"  {index_name(mult, pct)}")

    # for mult, pct in combos:
    #     print(f"\n{'='*80}")
    #     print(f"PROCESSING: {mult}x, {pct}%  →  {index_name(mult, pct)}")
    #     print(f"{'='*80}")
    #     try:
    #         if USE_CHUNK:
    #             process_chunks(mult, pct)
    #         else:
    #             process_single(mult, pct)
    #     except Exception as e:
    #         print(f"  ❌ Error ({mult}x_{pct}pct): {e}")
    #         import traceback; traceback.print_exc()

    # [통합 로직] 배치 전략 결정
    routing_config = {}
    if AUG_SHARD_DISTRIBUTION:
        # 분산 모드: 'docs' 속성 노드로 격리
        routing_config = {"box_type": ATTR_DOCS}

    for mult, pct in combos:
        try:
            use_chunk_for_this = (
                USE_CHUNK
                and pct == CHUNK_ONLY_PCT     # 100%만 chunk
                and mult >= CHUNK_THRESHOLD_MULT  # 20x/30x만 chunk
            )
            if use_chunk_for_this:
                process_chunks(mult, pct, routing_config)
            else:
                process_single(mult, pct, routing_config)
        except Exception as e:
            print(f"  ❌ Error ({mult}x_{pct}pct): {e}")
            import traceback; traceback.print_exc()

    elapsed = time.perf_counter() - start

    print(f"\n{'='*80}")
    print(f"✅ ALL DONE (V2.0)")
    print(f"{'='*80}")
    print(f"Multipliers     : {GAUSSIAN_AUG_MULTIPLIERS}")
    print(f"Coreset pcts    : {CORESET_PERCENTAGES}")
    print(f"Vec type        : {VEC_ELEMENT_TYPE} ({VEC_TYPE_LABEL})")
    print(f"Total time      : {elapsed:.2f}s")

    print(f"\n[verify] Final indices:")
    client.indices.refresh(index="fraud_ecom_aug_*")
    indices = client.cat.indices(index="fraud_ecom_aug_*", format="json")
    for idx in sorted(indices, key=lambda x: x['index']):
        docs = idx.get('docs.count', '0')
        size = idx.get('store.size', '?')
        print(f"  {idx['index']}: {docs} docs  ({size})")


if __name__ == "__main__":
    main()
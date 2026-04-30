"""
Elasticsearch Index Manager - V5.0
=====================================

인덱스 명명 규칙:
  Raw+Vec   : fraud_ecom_{EXPERIMENT_CASE}_percentage_{PCT}_cluster_tree_vec
              (experiment_case + coreset percentage별로 구분)
  Percolator: fraud_ecom_{EXPERIMENT_CASE}_{VERSION}_tree_rules_percolator
  Aug Vec   : fraud_ecom_aug_{MULT}x_{PCT}pct_{VEC_TYPE}_vec
              (gaussian augmentation 인덱스, float32 | int8)

Commands:
  list              : 전체 인덱스 목록 (tree vec / percolator / aug vec / other)
  inspect           : 인덱스 상세 (매핑, 통계, 샘플)
  test              : percolate 기본 동작 테스트
  compare-versions  : experiment_case별 v1~v8 percolate 정확도 비교
  compare           : experiment_case × version 인덱스 현황 매트릭스 + aug 현황
  delete-all        : fraud_ecom_* 인덱스 전체 삭제 (확인 필요)
  delete-vec        : raw+vec (tree) 인덱스만 삭제
  delete-aug        : aug vec 인덱스 삭제 (mult / pct / vec_type 선택 가능)
  delete-percolator : percolator 인덱스 삭제 (케이스별 / 버전별 선택 가능)

delete-aug 사용법:
  # 전체 aug 삭제
  python manage_indices.py delete-aug

  # 특정 multiplier만 삭제
  python manage_indices.py delete-aug --mult 10

  # 특정 multiplier + pct만 삭제
  python manage_indices.py delete-aug --mult 10 --pct 100

  # int8 인덱스만 삭제
  python manage_indices.py delete-aug --vec-type int8

  # dry-run
  python manage_indices.py delete-aug --mult 30 --dry-run

delete-percolator 사용법:
  python manage_indices.py delete-percolator
  python manage_indices.py delete-percolator --case pca_64
  python manage_indices.py delete-percolator --case pca_64 --version v4
  python manage_indices.py delete-percolator --case pca_64 --version v4 --dry-run
"""

import os
import re
import sys
import json
import argparse
from elasticsearch import Elasticsearch

try:
    import numpy as np
    import joblib
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

sys.path.append('/app')
try:
    from percolate_query_builder import get_query_builder, EXPERIMENT_CONFIG, QUERY_BUILDERS
    QUERY_BUILDER_AVAILABLE = True
except ImportError:
    QUERY_BUILDER_AVAILABLE = False
    EXPERIMENT_CONFIG = {
        "emb_vectors":  {"vector_dim": 576},
        "pca_32":       {"vector_dim": 32},
        "pca_64":       {"vector_dim": 64},
        "pca_128":      {"vector_dim": 128},
        "pca_256":      {"vector_dim": 256},
        "k50":          {"vector_dim": 576},
        "k100":         {"vector_dim": 576},
        "k200":         {"vector_dim": 576},
        "pca_64_k100":  {"vector_dim": 64},
        "pca_64_k200":  {"vector_dim": 64},
    }
    print("[warn] percolate_query_builder.py not found. Limited functionality.")

# ===========================
# Configuration
# ===========================

ES_URL          = os.getenv("ES_URL")
PASSWORD        = os.getenv("PASSWORD")
REQUEST_TIMEOUT = int(os.getenv("ES_REQUEST_TIMEOUT", "120"))
EXPERIMENT_CASE = os.getenv("EXPERIMENT_CASE", "pca_64")
PCA_MODEL_PATH  = os.getenv("PCA_MODEL_PATH", "/data/pca_model.joblib")

KNOWN_VERSIONS  = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11a","v11b","v11c","v12","v13","v14"]
KNOWN_VEC_TYPES = ["float32", "int8"]

client = Elasticsearch(
    [ES_URL],
    verify_certs=False,
    ssl_show_warn=False,
    basic_auth=("elastic", PASSWORD),
    request_timeout=REQUEST_TIMEOUT,
)
print(f"[es] Connected: {client.info()['cluster_name']}")

# ===========================
# Index Type Detectors
# ===========================

def is_vec_index(index_name: str) -> bool:
    """
    tree raw+vec 인덱스 여부.
    fraud_ecom_{experiment_case}_percentage_{PCT}_cluster_tree_vec
    """
    return "cluster_tree_vec" in index_name and "aug" not in index_name


def is_aug_index(index_name: str) -> bool:
    """
    Gaussian aug vec 인덱스 여부.
    fraud_ecom_aug_{MULT}x_{PCT}pct_{VEC_TYPE}_vec
    """
    return index_name.startswith("fraud_ecom_aug_")


def is_percolator_index(index_name: str) -> bool:
    return "tree_rules_percolator" in index_name


# ===========================
# Index Attribute Parsers
# ===========================

def detect_experiment_case(index_name: str) -> str | None:
    """
    percolator 인덱스 이름에서 experiment_case 추출.
    ★ 긴 이름 우선 매칭 (pca_64_k100 이 pca_64로 잘못 매칭되는 버그 방지)
    """
    for case in sorted(EXPERIMENT_CONFIG.keys(), key=len, reverse=True):
        if f"_{case}_" in index_name or index_name.endswith(f"_{case}"):
            return case
    return None


def detect_version(index_name: str) -> str | None:
    """인덱스 이름에서 query version(v1~v8) 추출."""
    for v in KNOWN_VERSIONS:
        if f"_{v}_" in index_name or index_name.endswith(f"_{v}"):
            return v
    return None


def detect_percentage(index_name: str) -> str | None:
    """
    인덱스 이름에서 coreset percentage 추출.
    fraud_ecom_percentage_100_cluster_tree_vec → "100"
    fraud_ecom_aug_10x_..._100_merged.npy     → "100"  (pct 필드)
    """
    m = re.search(r'percentage_(\d+)', index_name)
    if m:
        return m.group(1)
    # aug 인덱스: fraud_ecom_aug_{mult}x_{pct}pct_...
    m = re.search(r'_(\d+)pct_', index_name)
    return m.group(1) if m else None


def detect_aug_multiplier(index_name: str) -> str | None:
    """
    aug 인덱스에서 multiplier 추출.
    fraud_ecom_aug_10x_100pct_float32_vec → "10"
    """
    m = re.search(r'aug_(\d+)x_', index_name)
    return m.group(1) if m else None


def detect_aug_vec_type(index_name: str) -> str | None:
    """
    aug 인덱스에서 vec_type 추출.
    fraud_ecom_aug_10x_100pct_float32_vec → "float32"
    fraud_ecom_aug_10x_100pct_int8_vec    → "int8"
    """
    for vt in KNOWN_VEC_TYPES:
        if f"_{vt}_" in index_name or index_name.endswith(f"_{vt}"):
            return vt
    return None


# ===========================
# PCA Helpers
# ===========================

_pca_model_cache = {}

def _get_pca_model(experiment_case: str):
    if experiment_case in _pca_model_cache:
        return _pca_model_cache[experiment_case]
    if not NUMPY_AVAILABLE:
        return None
    if os.path.exists(PCA_MODEL_PATH):
        model = joblib.load(PCA_MODEL_PATH)
        _pca_model_cache[experiment_case] = model
        print(f"[pca] Loaded: {PCA_MODEL_PATH}")
        return model
    print(f"[pca] Not found: {PCA_MODEL_PATH}")
    _pca_model_cache[experiment_case] = None
    return None


def _is_pca_case(experiment_case: str) -> bool:
    return EXPERIMENT_CONFIG.get(experiment_case, {}).get("vector_dim", 576) < 576


def build_percolate_doc_from_vec(doc: dict, experiment_case: str = None) -> dict:
    """
    raw+vec 인덱스 문서(vec: 576-dim)에서 percolate test document 생성.
    - PCA 케이스    : vec(576) → PCA transform → {v1~vN}
    - non-PCA 케이스: vec(576) → {v1~v576} 직접 매핑
    """
    ec  = experiment_case or EXPERIMENT_CASE
    vec = doc.get("vec")

    if not vec or not isinstance(vec, list):
        return {k: v for k, v in doc.items()
                if k.startswith("v") and k[1:].isdigit()}

    if _is_pca_case(ec) and NUMPY_AVAILABLE:
        pca = _get_pca_model(ec)
        if pca is not None:
            vec_np      = np.array(vec, dtype=np.float32).reshape(1, -1)
            transformed = pca.transform(vec_np).flatten()
            return {f"v{i+1}": float(v) for i, v in enumerate(transformed)}
        else:
            print(f"[warn] PCA model unavailable for '{ec}' — using raw vec (accuracy reduced)")

    return {f"v{i+1}": float(v) for i, v in enumerate(vec)}


# ===========================
# list_indices
# ===========================

def list_indices(pattern: str = "fraud_ecom_*"):
    """
    패턴에 매칭되는 인덱스 목록 출력.
    tree vec / percolator / aug vec / other 4개 섹션으로 분류.
    """
    indices = client.cat.indices(index=pattern, format="json")
    if not indices:
        print(f"No indices matching: {pattern}")
        return

    vec_indices  = []
    aug_indices  = []
    perc_indices = []
    other_indices = []

    for idx in sorted(indices, key=lambda x: x['index']):
        name = idx['index']
        if is_aug_index(name):
            aug_indices.append(idx)
        elif is_vec_index(name):
            vec_indices.append(idx)
        elif is_percolator_index(name):
            perc_indices.append(idx)
        else:
            other_indices.append(idx)

    W = 68
    print(f"\n{'='*100}")
    print(f"Indices matching '{pattern}':")
    print(f"{'='*100}")
    print(f"{'Index':<{W}} {'Docs':>8} {'Size':>8}  {'Type':<14} {'Detail'}")
    print(f"{'-'*100}")

    def _print_row(idx, itype, extra):
        name = idx['index']
        docs = idx.get('docs.count', '0')
        size = idx.get('store.size', '0b')
        print(f"{name:<{W}} {docs:>8} {size:>8}  {itype:<14} {extra}")

    if vec_indices:
        print("  ── Tree Raw+Vec fraud_ecom_{EXPERIMENT_CASE}_percentage_{PCT}_cluster_tree_vec ──")
        for idx in vec_indices:
            pct = detect_percentage(idx['index']) or "?"
            _print_row(idx, "tree-vec", f"pct={pct}%")

    if perc_indices:
        print("  ── Percolator  (fraud_ecom_{CASE}_{VER}_tree_rules_percolator) ──")
        for idx in perc_indices:
            case = detect_experiment_case(idx['index']) or "?"
            ver  = detect_version(idx['index']) or "?"
            _print_row(idx, "percolator", f"{case}  /  {ver}")

    if aug_indices:
        print("  ── Gaussian Aug Vec  (fraud_ecom_aug_{MULT}x_{PCT}pct_{TYPE}_vec) ──")
        for idx in aug_indices:
            mult = detect_aug_multiplier(idx['index']) or "?"
            pct  = detect_percentage(idx['index']) or "?"
            vt   = detect_aug_vec_type(idx['index']) or "?"
            _print_row(idx, "aug-vec", f"mult={mult}x  pct={pct}%  type={vt}")

    if other_indices:
        print("  ── Other ──")
        for idx in other_indices:
            _print_row(idx, "other", "")

    print(f"{'='*100}")
    print(f"Total: {len(indices)} indices  "
          f"(tree-vec={len(vec_indices)}, percolator={len(perc_indices)}, "
          f"aug-vec={len(aug_indices)}, other={len(other_indices)})")
    total_docs = sum(int(i.get('docs.count', 0)) for i in indices)
    print(f"Total docs: {total_docs:,}")


# ===========================
# inspect_index
# ===========================

def inspect_index(index_name: str):
    """인덱스 상세 정보 (매핑, 설정, 통계, 샘플)."""
    percentage = detect_percentage(index_name)

    print(f"\n{'='*80}")
    print(f"INDEX INSPECTION: {index_name}")

    if is_aug_index(index_name):
        mult = detect_aug_multiplier(index_name) or "?"
        pct  = detect_percentage(index_name) or "?"
        vt   = detect_aug_vec_type(index_name) or "?"
        print(f"Type               : aug-vec  (mult={mult}x, pct={pct}%, vec_type={vt})")
    elif is_vec_index(index_name):
        print(f"Type               : tree raw+vec  (pct={percentage or '?'}%)")
    elif is_percolator_index(index_name):
        ec   = detect_experiment_case(index_name)
        ver  = detect_version(index_name)
        vdim = EXPERIMENT_CONFIG.get(ec, {}).get("vector_dim", "?") if ec else "?"
        print(f"Type               : percolator")
        print(f"Experiment case    : {ec or '?'}  (vector_dim={vdim})")
        print(f"Version            : {ver or '?'}")
    print(f"{'='*80}")

    # ── 1. Mapping ──────────────────────────────────────────────
    print(f"\n[1] Mapping")
    try:
        mapping = client.indices.get_mapping(index=index_name)
        props   = mapping[index_name]['mappings'].get('properties', {})
        print(f"  Total fields: {len(props)}")
        field_types: dict = {}
        for field, spec in props.items():
            ftype = spec.get('type', 'object')
            field_types.setdefault(ftype, []).append(field)
        for ftype, fields in sorted(field_types.items()):
            print(f"    {ftype}: {len(fields)} fields", end="")
            if len(fields) <= 5:
                print(f"  → {fields}")
            else:
                print(f"  → {fields[:5]}...")
        if 'query' in props:
            print(f"  ✅ Percolator index (has 'query' field)")
        if 'vec' in props:
            dims = props['vec'].get('dims', '?')
            vidx = props['vec'].get('index', True)
            print(f"  ✅ Vector field: vec ({dims} dims, index={vidx})")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # ── 2. Stats ─────────────────────────────────────────────────
    print(f"\n[2] Stats")
    try:
        count = client.count(index=index_name)["count"]
        stats = client.indices.stats(index=index_name)
        store = stats['_all']['total']['store']['size_in_bytes']
        print(f"  Docs       : {count:,}")
        print(f"  Store size : {store:,} bytes ({store / 1024 / 1024:.2f} MB)")
        if count > 0:
            print(f"  Avg doc    : {store / count:.0f} bytes")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # ── 3. Sample ────────────────────────────────────────────────
    print(f"\n[3] Sample Documents (3)")
    try:
        result = client.search(
            index=index_name,
            body={"query": {"match_all": {}}, "size": 3}
        )
        hits = result["hits"]["hits"]
        print(f"  Total: {result['hits']['total']['value']:,} docs")
        for i, hit in enumerate(hits, 1):
            doc = hit["_source"]
            print(f"\n  [{i}] _id: {hit['_id']}")
            if 'cluster_id' in doc:
                print(f"      cluster_id : {doc['cluster_id']}")
            if 'original_index' in doc:
                print(f"      original_index : {doc['original_index']}")
            if 'query' in doc:
                q    = doc['query'].get('bool', {})
                ncls = len(q.get('filter', q.get('should', q.get('must', []))))
                print(f"      query      : percolator ({ncls} conditions)")
            if 'vec' in doc:
                print(f"      vec        : {len(doc['vec'])} dims")
            other = [k for k in doc if k not in ('cluster_id', 'original_index', 'query', 'vec', 'persona')]
            print(f"      other      : {other[:6]}...")
    except Exception as e:
        print(f"  ❌ Error: {e}")


# ===========================
# test_percolate
# ===========================

def test_percolate(percolator_index: str, raw_vec_index: str = None,
                   num_samples: int = 5):
    """
    tree raw+vec 인덱스 샘플 문서로 percolate 기본 동작 테스트.
    raw_vec_index 미지정 시 percentage_100 인덱스 자동 사용.
    """
    ec      = detect_experiment_case(percolator_index) or EXPERIMENT_CASE
    raw_idx = raw_vec_index or "fraud_ecom_{ec}_percentage_100_cluster_tree_vec"

    print(f"\n{'='*80}")
    print(f"PERCOLATE TEST")
    print(f"  Percolator : {percolator_index}")
    print(f"  Raw+Vec    : {raw_idx}")
    print(f"  Case       : {ec}")
    print(f"{'='*80}")

    try:
        result  = client.search(
            index=raw_idx,
            body={"query": {"match_all": {}}, "size": num_samples}
        )
        samples = result["hits"]["hits"]
        print(f"\nTest docs: {len(samples)}\n")
    except Exception as e:
        print(f"❌ Cannot load samples: {e}")
        return

    correct = total = 0
    for i, hit in enumerate(samples, 1):
        src            = hit["_source"]
        doc_id         = hit["_id"]
        actual_cluster = src.get('cluster_id')
        perc_doc       = build_percolate_doc_from_vec(src, ec)

        pr      = client.search(
            index=percolator_index,
            body={
                "query": {"percolate": {"field": "query", "document": perc_doc}},
                "size": 5
            }
        )
        matches = pr["hits"]["total"]["value"]
        total  += 1

        print(f"[{i}] id={doc_id}  actual_cluster={actual_cluster}  matches={matches}")
        if matches > 0:
            for m in pr["hits"]["hits"][:3]:
                mc = m["_source"]["cluster_id"]
                ok = "✅" if mc == actual_cluster else "❌"
                print(f"     {ok} cluster={mc}  score={m['_score']:.4f}")
                if mc == actual_cluster:
                    correct += 1
                    break
        else:
            print(f"     ❌ No matches")
        print()

    if total > 0:
        print(f"Top-1 Accuracy: {correct}/{total} = {correct/total*100:.1f}%")


# ===========================
# compare_versions
# ===========================

def compare_versions(experiment_case: str = None, raw_vec_index: str = None,
                     num_samples: int = 20):
    """
    같은 experiment_case의 v1~v8 percolate 정확도 비교.
    raw_vec_index 미지정 시 percentage_100 인덱스 자동 사용.
    """
    ec      = experiment_case or EXPERIMENT_CASE
    raw_idx = raw_vec_index or "fraud_ecom_{ec}_percentage_100_cluster_tree_vec"

    print(f"\n{'='*70}")
    print(f"VERSION COMPARISON  experiment_case={ec}")
    print(f"  raw+vec index : {raw_idx}")
    print(f"  samples       : {num_samples}")
    print(f"{'='*70}")

    try:
        result  = client.search(
            index=raw_idx,
            body={"query": {"match_all": {}}, "size": num_samples}
        )
        samples = result["hits"]["hits"]
        print(f"Test docs: {len(samples)} from {raw_idx}\n")
    except Exception as e:
        print(f"❌ Cannot load samples: {e}")
        return

    print(f"{'Version':<8} {'Builder':<28} {'Avg Matches':>12} {'Top-1 Acc':>10} {'Avg Score':>10}")
    print(f"{'-'*70}")

    for version in KNOWN_VERSIONS:
        perc_idx = f"fraud_ecom_{ec}_{version}_tree_rules_percolator"
        if not client.indices.exists(index=perc_idx):
            print(f"{version:<8} {'(index not found)':<28}")
            continue

        builder_name = QUERY_BUILDERS[version].__name__ if QUERY_BUILDER_AVAILABLE else "-"
        correct = total = matched_sum = 0
        score_sum = 0.0

        for hit in samples:
            src            = hit["_source"]
            actual_cluster = src.get('cluster_id')
            perc_doc       = build_percolate_doc_from_vec(src, ec)
            if not perc_doc:
                continue
            try:
                pr   = client.search(
                    index=perc_idx,
                    body={
                        "query": {"percolate": {"field": "query", "document": perc_doc}},
                        "size": 1,
                        "_source": ["cluster_id"]
                    }
                )
                hits = pr["hits"]["hits"]
                total       += 1
                matched_sum += pr["hits"]["total"]["value"]
                if hits:
                    score_sum += hits[0].get("_score", 0.0)
                    if hits[0]["_source"]["cluster_id"] == actual_cluster:
                        correct += 1
            except Exception:
                pass

        acc       = f"{correct/total*100:.1f}%" if total else "-"
        avg_score = f"{score_sum/total:.3f}"    if total else "-"
        avg_match = f"{matched_sum/total:.1f}"  if total else "-"
        print(f"{version:<8} {builder_name:<28} {avg_match:>12} {acc:>10} {avg_score:>10}")

    print(f"{'='*70}")


# ===========================
# compare_cases
# ===========================

def compare_cases(pattern: str = "fraud_ecom_*_tree_rules_percolator"):
    """
    experiment_case × version 인덱스 현황 매트릭스 출력.
    + aug vec 인덱스 현황 (mult × pct × vec_type)
    """
    # ── Percolator 매트릭스 ───────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"INDEX MATRIX — Percolator  (pattern: {pattern})")
    print(f"{'='*80}")

    indices = client.cat.indices(index=pattern, format="json")
    if indices:
        matrix: dict = {}
        for idx in indices:
            name = idx['index']
            if not is_percolator_index(name):
                continue
            case = detect_experiment_case(name)
            ver  = detect_version(name)
            docs = int(idx.get('docs.count', 0))
            if case and ver:
                matrix.setdefault(case, {})[ver] = docs

        header = f"{'Case':<18}" + "".join(f"  {v:>10}" for v in KNOWN_VERSIONS)
        print(header)
        print("-" * len(header))
        for case in sorted(matrix):
            row = f"{case:<18}"
            for v in KNOWN_VERSIONS:
                docs = matrix[case].get(v)
                row += f"  {docs:>10,}" if docs is not None else f"  {'(missing)':>10}"
            print(row)
        print(f"\nTotal percolator indices: {len(indices)}")
    else:
        print("No percolator indices found.")

    # ── Tree Raw+Vec 현황 ─────────────────────────────────────────
    vec_indices = client.cat.indices(
        index="fraud_ecom_*_percentage_*_cluster_tree_vec", format="json"
    )
    if vec_indices:
        print(f"\n{'─'*70}")
        print("Tree Raw+Vec Indices  (experiment_case별):")
        print(f"  {'Index':<52} {'Docs':>10}  {'Size':>8}")
        for idx in sorted(vec_indices, key=lambda x: x['index']):
            pct  = detect_percentage(idx['index']) or "?"
            docs = int(idx.get('docs.count', 0))
            size = idx.get('store.size', '0b')
            print(f"  {idx['index']:<52} {docs:>10,}  {size:>8}  (pct={pct}%)")

    # ── Aug Vec 현황 ─────────────────────────────────────────────
    aug_indices = client.cat.indices(index="fraud_ecom_aug_*", format="json")
    if aug_indices:
        print(f"\n{'─'*70}")
        print("Gaussian Aug Vec Indices  (fraud_ecom_aug_{{MULT}}x_{{PCT}}pct_{{TYPE}}_vec):")
        print(f"  {'Index':<58} {'Docs':>10}  {'Size':>8}")

        # mult × pct × vec_type 매트릭스 요약
        aug_matrix: dict = {}   # (mult, pct, vec_type) → docs
        for idx in sorted(aug_indices, key=lambda x: x['index']):
            name = idx['index']
            mult = detect_aug_multiplier(name) or "?"
            pct  = detect_percentage(name) or "?"
            vt   = detect_aug_vec_type(name) or "?"
            docs = int(idx.get('docs.count', 0))
            size = idx.get('store.size', '0b')
            aug_matrix[(mult, pct, vt)] = docs
            print(f"  {name:<58} {docs:>10,}  {size:>8}  (mult={mult}x, pct={pct}%, type={vt})")

        print(f"\n  총 aug 인덱스: {len(aug_indices)}")
        mults = sorted({k[0] for k in aug_matrix}, key=lambda x: int(x) if x.isdigit() else 0)
        print(f"  Multipliers  : {mults}")
        vts   = sorted({k[2] for k in aug_matrix})
        print(f"  Vec types    : {vts}")
    else:
        print(f"\n{'─'*70}")
        print("Aug Vec Indices: 없음  (ingest_gaussian_aug.py 미실행 또는 삭제됨)")


# ===========================
# Delete Helpers
# ===========================

def _print_delete_targets(indices: list, dry_run: bool = False):
    total_docs = sum(int(i.get('docs.count', 0)) for i in indices)
    tag = "[dry-run] " if dry_run else ""
    print(f"\n{tag}⚠️  삭제 대상: {len(indices)}개 인덱스, 총 {total_docs:,} docs")
    for idx in sorted(indices, key=lambda x: x['index']):
        docs = idx.get('docs.count', '0')
        size = idx.get('store.size', '?')
        print(f"  - {idx['index']}  ({docs} docs, {size})")


def _confirm_and_delete(indices: list, dry_run: bool = False):
    if not indices:
        print("삭제 대상 인덱스 없음.")
        return

    _print_delete_targets(indices, dry_run=dry_run)

    if dry_run:
        print("\n[dry-run] 실제 삭제 안 함. --dry-run 제거 후 재실행하세요.")
        return

    ans = input("\n정말 삭제하시겠습니까? (yes 입력): ").strip()
    if ans != "yes":
        print("취소됨.")
        return

    deleted = failed = 0
    for idx in indices:
        name = idx['index']
        try:
            client.indices.delete(index=name)
            print(f"  ✅ Deleted: {name}")
            deleted += 1
        except Exception as e:
            print(f"  ❌ Failed: {name} — {e}")
            failed += 1

    print(f"\n삭제 완료: {deleted} deleted, {failed} failed")


# ===========================
# Delete Commands
# ===========================

def delete_all(dry_run: bool = False):
    """모든 fraud_ecom_* 인덱스 삭제 (tree vec + percolator + aug vec)."""
    print(f"\n{'='*70}")
    print("DELETE ALL:  fraud_ecom_*")
    print("  ⚠️  tree vec + percolator + aug vec 전체 삭제")
    print(f"{'='*70}")
    indices = client.cat.indices(index="fraud_ecom_*", format="json")
    _confirm_and_delete(indices or [], dry_run=dry_run)


# 변경
# def delete_vec_indices(experiment_case: str = None, dry_run: bool = False):
def delete_vec_indices(experiment_case: str = None, pct: str = None, dry_run: bool = False):
    """
    Tree Raw+Vec 인덱스만 삭제.
    fraud_ecom_{experiment_case}_percentage_*_cluster_tree_vec

    Args:
        experiment_case: 특정 케이스만 삭제 (None이면 전체)
        dry_run: True면 대상 목록만 출력
    """
    print(f"\n{'='*70}")
    # if experiment_case:
    #     pattern = f"fraud_ecom_{experiment_case}_percentage_*_cluster_tree_vec"
    #     print(f"DELETE TREE VEC (case={experiment_case}):  {pattern}")
    # else:
    #     pattern = "fraud_ecom_*_percentage_*_cluster_tree_vec"
    #     print(f"DELETE TREE VEC (all cases):  {pattern}")
    pct_part = pct if pct else "*"
    if experiment_case:
        pattern = f"fraud_ecom_{experiment_case}_percentage_{pct_part}_cluster_tree_vec"
        print(f"DELETE TREE VEC (case={experiment_case}, pct={pct or 'all'}):  {pattern}")
    else:
        pattern = f"fraud_ecom_*_percentage_{pct_part}_cluster_tree_vec"
        print(f"DELETE TREE VEC (all cases, pct={pct or 'all'}):  {pattern}")
    print(f"{'='*70}")
    indices = client.cat.indices(index=pattern, format="json")
    indices = [i for i in (indices or []) if is_vec_index(i['index'])]
    _confirm_and_delete(indices, dry_run=dry_run)


def delete_aug_indices(
    mult: str = None,
    pct: str = None,
    vec_type: str = None,
    dry_run: bool = False,
):
    """
    Gaussian Aug Vec 인덱스 삭제.

    조합:
      (None, None, None) → 전체 aug 삭제
      (mult=10, None, None) → 10x 전체 삭제
      (mult=10, pct=100, None) → 10x + 100% 삭제
      (None, None, vec_type=int8) → int8 전체 삭제

    Args:
        mult:     multiplier 숫자 문자열 (예: "10")
        pct:      coreset percentage (예: "100")
        vec_type: "float32" | "int8"
        dry_run:  True면 대상 목록만 출력
    """
    print(f"\n{'='*70}")
    print("DELETE AUG VEC")

    # 패턴 구성
    mult_part = f"{mult}x" if mult else "*"
    pct_part  = f"{pct}pct" if pct else "*pct"
    vt_part   = vec_type if vec_type else "*"
    pattern   = f"fraud_ecom_aug_{mult_part}_{pct_part}_{vt_part}_vec"

    print(f"  Pattern  : {pattern}")
    if mult:     print(f"  Mult     : {mult}x")
    if pct:      print(f"  Pct      : {pct}%")
    if vec_type: print(f"  Vec type : {vec_type}")
    print(f"{'='*70}")

    indices = client.cat.indices(index=pattern, format="json")
    indices = [i for i in (indices or []) if is_aug_index(i['index'])]

    if not indices:
        print("삭제 대상 aug 인덱스 없음.")
        return

    mults_affected = sorted({detect_aug_multiplier(i['index']) or "?" for i in indices},
                             key=lambda x: int(x) if x.isdigit() else 0)
    vts_affected   = sorted({detect_aug_vec_type(i['index']) or "?" for i in indices})
    print(f"\n  영향 multipliers : {mults_affected}")
    print(f"  영향 vec types   : {vts_affected}")

    _confirm_and_delete(indices, dry_run=dry_run)


def delete_percolator_indices(
    experiment_case: str = None,
    version: str = None,
    dry_run: bool = False,
):
    """
    Percolator 인덱스 삭제.

    조합:
      (None, None)         → 전체 percolator 삭제
      (case=pca_64, None)  → pca_64의 모든 버전 삭제
      (case=pca_64, v=v4)  → pca_64의 v4만 삭제
      (None, v=v4)         → 모든 케이스의 v4 삭제
    """
    print(f"\n{'='*70}")
    print("DELETE PERCOLATOR")

    if experiment_case and version:
        pattern = f"fraud_ecom_{experiment_case}_{version}_tree_rules_percolator"
        print(f"  Mode   : 특정 케이스 + 특정 버전")
    elif experiment_case:
        pattern = f"fraud_ecom_{experiment_case}_*_tree_rules_percolator"
        print(f"  Mode   : 특정 케이스 전체 버전")
    elif version:
        pattern = f"fraud_ecom_*_{version}_tree_rules_percolator"
        print(f"  Mode   : 모든 케이스의 특정 버전")
    else:
        pattern = "fraud_ecom_*_tree_rules_percolator"
        print(f"  Mode   : 전체 percolator 삭제")

    print(f"  Pattern: {pattern}")
    print(f"{'='*70}")

    indices = client.cat.indices(index=pattern, format="json")
    indices = [i for i in (indices or []) if is_percolator_index(i['index'])]

    if not indices:
        print("삭제 대상 percolator 인덱스 없음.")
        return

    affected_cases    = sorted({detect_experiment_case(i['index']) or "?" for i in indices})
    affected_versions = sorted({detect_version(i['index']) or "?" for i in indices})
    print(f"\n  영향 케이스  : {affected_cases}")
    print(f"  영향 버전    : {affected_versions}")

    _confirm_and_delete(indices, dry_run=dry_run)


def check_shard_distribution(pattern: str):
    """
    Elasticsearch 인덱스 샤드 배치 현황 출력 (Distributed 모드 검증용)
    """
    print(f"\n{'='*100}")
    print(f" SHARD DISTRIBUTION CHECK (Pattern: {pattern})")
    print(f"{'='*100}")

    try:
        # h(header) 인자에 box_type 속성을 포함하여 노드 속성까지 확인
        shards = client.cat.shards(
            index=pattern, 
            format="json", 
            h="index,shard,prirep,state,node,box_type"
        )
        
        if not shards:
            print(f" ❌ No shards found for pattern '{pattern}'")
            return

        # 출력 헤더
        header = f"{'INDEX':<55} {'SHARD':<5} {'TYPE':<4} {'STATE':<12} {'NODE'}"
        print(header)
        print("-" * 100)

        for s in shards:
            idx_name = s['index']
            shard_num = s['shard']
            p_or_r = s['prirep'] # p: primary, r: replica
            state = s['state']
            node = s['node']
            
            print(f"{idx_name:<55} {shard_num:<5} {p_or_r:<4} {state:<12} {node}")
            
    except Exception as e:
        print(f" ❌ Error checking shards: {e}")

# ===========================
# CLI
# ===========================

def main():
    parser = argparse.ArgumentParser(
        description="ES Index Manager V5.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 전체 인덱스 목록 (tree vec / percolator / aug vec)
  python manage_indices.py list

  # aug 인덱스만 필터
  python manage_indices.py list --pattern 'fraud_ecom_aug_*'

  # 인덱스 상세
  python manage_indices.py inspect -i fraud_ecom_pca_64_k100_v2_tree_rules_percolator
  python manage_indices.py inspect -i fraud_ecom_aug_10x_100pct_float32_vec

  # percolate 테스트 (raw+vec 자동 사용)
  python manage_indices.py test -i fraud_ecom_pca_64_k100_v2_tree_rules_percolator

  # v1~v8 버전 비교
  python manage_indices.py compare-versions --case pca_64_k100 -n 50

  # 전체 현황 매트릭스 (percolator + aug)
  python manage_indices.py compare

  # percolate quuery shard 비교
  python /app/manage_indices.py --check-shards --pattern 'fraud_ecom_percolator*'"

  # ── 삭제 ──────────────────────────────────────────────
  # aug 전체 삭제 (dry-run)
  python manage_indices.py delete-aug --dry-run

  # 특정 multiplier만 삭제
  python manage_indices.py delete-aug --mult 10 --dry-run
  python manage_indices.py delete-aug --mult 10

  # multiplier + pct 조합 삭제
  python manage_indices.py delete-aug --mult 10 --pct 100

  # int8 인덱스만 삭제
  python manage_indices.py delete-aug --vec-type int8

  # percolator 전체
  python manage_indices.py delete-percolator --dry-run
  python manage_indices.py delete-percolator

  # 특정 케이스 + 버전
  python manage_indices.py delete-percolator --case pca_64_k100 --version v4

  # tree raw+vec 삭제
  python manage_indices.py delete-vec --dry-run

  # 전체 삭제
  python manage_indices.py delete-all --dry-run
"""
    )

    parser.add_argument(
        "command",
        choices=[
            "list", "inspect", "test",
            "compare-versions", "compare",
            "delete-all", "delete-vec", "delete-aug", "delete-percolator"
        ],
        help="Command to execute"
    )
    parser.add_argument("--index",    "-i", help="인덱스 이름 (inspect / test)")
    parser.add_argument("--raw-index","-r", help="Raw+Vec 인덱스 (default: percentage_100)")
    parser.add_argument("--case",     "-c", help="experiment_case (e.g. pca_64_k100)")
    parser.add_argument("--version",  "-v", help="버전 (e.g. v1~v8)")
    parser.add_argument("--pattern",  "-p", default="fraud_ecom_*")
    parser.add_argument("--samples",  "-n", type=int, default=20)
    # delete-aug 전용
    parser.add_argument("--mult",     type=str, help="Aug multiplier 숫자 (e.g. 10)")
    parser.add_argument("--pct",      type=str, help="Coreset pct (e.g. 100)")
    parser.add_argument("--vec-type", type=str, choices=KNOWN_VEC_TYPES,
                        help="Aug vec type: float32 | int8")
    parser.add_argument("--dry-run",  action="store_true",
                        help="삭제 명령: 실제 삭제 없이 대상 목록만 출력")

    parser.add_argument("--check-shards", action="store_true", help="인덱스 샤드 분산 현황 확인 (Distributed 모드 검증)")


    args = parser.parse_args()

    try:
        if args.command == "list":
            list_indices(args.pattern)

        elif args.command == "inspect":
            if not args.index:
                print("❌ --index required"); return
            inspect_index(args.index)

        elif args.command == "test":
            if not args.index:
                print("❌ --index required"); return
            test_percolate(args.index, args.raw_index, args.samples)

        elif args.command == "compare-versions":
            compare_versions(
                experiment_case=args.case,
                raw_vec_index=args.raw_index,
                num_samples=args.samples,
            )

        elif args.command == "compare":
            compare_cases(args.pattern)

        elif args.command == "delete-all":
            delete_all(dry_run=args.dry_run)
            
        elif args.command == "delete-vec":
            # delete_vec_indices(experiment_case=args.case, dry_run=args.dry_run)
            delete_vec_indices(experiment_case=args.case, pct=args.pct, dry_run=args.dry_run)

        elif args.command == "delete-aug":
            delete_aug_indices(
                mult=args.mult,
                pct=args.pct,
                vec_type=args.vec_type,
                dry_run=args.dry_run,
            )

        elif args.command == "delete-percolator":
            if args.version and args.version not in KNOWN_VERSIONS:
                print(f"❌ Unknown version '{args.version}'. Valid: {KNOWN_VERSIONS}")
                return
            delete_percolator_indices(
                experiment_case=args.case,
                version=args.version,
                dry_run=args.dry_run,
            )
        # [추가] check-shards 플래그가 들어온 경우 가장 먼저 실행
        elif args.check_shards:
            # --pattern 인자가 있으면 해당 패턴을 사용하고, 없으면 기본값 사용
            pattern = args.pattern if args.pattern else "fraud_ecom_*"
            check_shard_distribution(pattern)
            return  # 실행 후 종료

    except KeyboardInterrupt:
        print("\n❌ Interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()

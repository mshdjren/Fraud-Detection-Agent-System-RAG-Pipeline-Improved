"""
vector_db_sweep.py — Vector DB 튜닝 (HNSW sweep + int8/BBQ quantization)
=========================================================================
PASTED 문서 1의 ② 구현. ES 유지, 기존 float32 소스 인덱스를 reindex하여
(m × ef_construction × quantization) 변형 인덱스를 만들고, 각 설정에서
num_candidates(ef_search) sweep으로 recall@k / latency P50·P99 / 인덱스
메모리를 측정한다. 산출물은 "operating point 선택" 근거인 트레이드오프
곡선(파레토 프론트)용 CSV + PNG.

측정 정의:
  recall@k : 동일 인덱스의 exact script_score KNN(l2norm) top-k를
             ground-truth 이웃으로 삼아, HNSW top-k와의 교집합 비율.
  latency  : ES took(ms) 대신 client wall-clock — retriever 관점의 실측.
             P50/P99, warm-up 5회 제외.
  memory   : _disk_usage(analyzed) 의 knn_vectors + _stats store size.

Usage:
  ES_URL=https://... PASSWORD=... \
  python vector_db_sweep.py --source-index fraud_ecom_pca_64_k100_..._vec \
      --n-queries 200 --k 5 \
      --m-list 16 32 --efc-list 100 200 --nc-list 20 50 100 200 \
      --quant-list float32 int8_hnsw bbq_hnsw \
      --output-dir ./results/vector_db_sweep
"""
from __future__ import annotations
import argparse
import json
import os
import time
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
from elasticsearch import Elasticsearch

ES_URL   = os.getenv("ES_URL")
PASSWORD = os.getenv("PASSWORD")

client = Elasticsearch([ES_URL], verify_certs=False, ssl_show_warn=False,
                       basic_auth=("elastic", PASSWORD), request_timeout=120)


# ============================================================
# 1. 변형 인덱스 생성 (reindex)
# ============================================================

def variant_name(src: str, m: int, efc: int, quant: str) -> str:
    return f"{src}_sweep_m{m}_efc{efc}_{quant}"


def create_variant(src: str, m: int, efc: int, quant: str, dims: int) -> str:
    """
    소스 인덱스에서 vec 필드만 index_options를 바꿔 reindex.
    quant: float32 → type hnsw / int8_hnsw / bbq_hnsw (ES 8.14+ / 8.18+).
    이미 존재하면 재사용 (idempotent — sweep 재실행 비용 절감).
    """
    dst = variant_name(src, m, efc, quant)
    if client.indices.exists(index=dst):
        print(f"  [skip] {dst} 이미 존재")
        return dst

    src_map = client.indices.get_mapping(index=src)[src]["mappings"]
    props = src_map.get("properties", {})
    hnsw_type = {"float32": "hnsw", "int8_hnsw": "int8_hnsw",
                 "bbq_hnsw": "bbq_hnsw"}[quant]
    props["vec"] = {
        "type": "dense_vector", "dims": dims, "index": True,
        "similarity": "l2_norm",
        "index_options": {"type": hnsw_type, "m": m, "ef_construction": efc},
    }
    client.indices.create(index=dst, body={
        "settings": {"number_of_shards": 1, "number_of_replicas": 0,
                     "refresh_interval": "-1"},
        "mappings": {"properties": props},
    })
    print(f"  [reindex] {src} → {dst} ...")
    client.reindex(body={"source": {"index": src}, "dest": {"index": dst}},
                   wait_for_completion=True, request_timeout=3600)
    client.indices.put_settings(index=dst, body={"refresh_interval": "1s"})
    client.indices.refresh(index=dst)
    client.indices.forcemerge(index=dst, max_num_segments=1,
                              request_timeout=3600)
    return dst


# ============================================================
# 2. Ground-truth 이웃 (exact KNN) — 쿼리 벡터도 인덱스에서 샘플링
# ============================================================

def sample_query_vectors(index: str, n: int, seed: int = 42) -> List[List[float]]:
    """인덱스 문서에서 vec을 n개 샘플 — 별도 test 임베딩 의존 없이 자립."""
    resp = client.search(index=index, body={
        "size": n, "query": {"function_score": {
            "query": {"match_all": {}},
            "random_score": {"seed": seed, "field": "_seq_no"}}},
        "_source": False, "docvalue_fields": [],
        "fields": [], "stored_fields": "_none_",
    })
    ids = [h["_id"] for h in resp["hits"]["hits"]]
    vecs = []
    for _id in ids:
        doc = client.get(index=index, id=_id, source_includes=["vec"])
        v = doc["_source"].get("vec")
        if v:
            vecs.append(v)
    return vecs


def exact_topk(index: str, vec: List[float], k: int) -> List[str]:
    body = {
        "size": k,
        "query": {"script_score": {
            "query": {"match_all": {}},
            "script": {"source": "1 / (1 + l2norm(params.q, 'vec'))",
                       "params": {"q": vec}}}},
        "_source": False,
    }
    resp = client.search(index=index, body=body)
    return [h["_id"] for h in resp["hits"]["hits"]]


def hnsw_topk(index: str, vec: List[float], k: int, num_candidates: int) -> List[str]:
    body = {"size": k,
            "knn": {"field": "vec", "query_vector": vec,
                    "k": k, "num_candidates": num_candidates},
            "_source": False}
    resp = client.search(index=index, body=body)
    return [h["_id"] for h in resp["hits"]["hits"]]


# ============================================================
# 3. 측정
# ============================================================

def measure_memory(index: str) -> Dict:
    out = {"store_mb": None, "knn_vectors_mb": None}
    try:
        stats = client.indices.stats(index=index, metric="store")
        out["store_mb"] = round(
            stats["indices"][index]["total"]["store"]["size_in_bytes"] / 1024**2, 2)
    except Exception:
        pass
    try:
        du = client.perform_request(
            "POST", f"/{index}/_disk_usage", params={"run_expensive_tasks": "true"})
        fields = du[index]["fields"]
        if "vec" in fields:
            out["knn_vectors_mb"] = round(
                fields["vec"].get("knn_vectors_in_bytes", 0) / 1024**2, 2)
    except Exception:
        pass
    return out


def measure_config(index: str, queries: List[List[float]],
                   gt: List[List[str]], k: int, num_candidates: int,
                   n_warmup: int = 5) -> Dict:
    recalls, lats = [], []
    for i, (vec, gt_ids) in enumerate(zip(queries, gt)):
        t0 = time.perf_counter()
        ids = hnsw_topk(index, vec, k, num_candidates)
        dt = (time.perf_counter() - t0) * 1000
        if i >= n_warmup:                       # warm-up 제외
            lats.append(dt)
        recalls.append(len(set(ids) & set(gt_ids)) / max(len(gt_ids), 1))
    lat = np.array(lats) if lats else np.array([0.0])
    return {
        "recall_at_k": round(float(np.mean(recalls)), 4),
        "latency_p50_ms": round(float(np.percentile(lat, 50)), 2),
        "latency_p99_ms": round(float(np.percentile(lat, 99)), 2),
        "n_queries": len(queries) - n_warmup,
    }


# ============================================================
# 4. Sweep 본체 + 파레토 산출
# ============================================================

def pareto_front(rows: List[Dict]) -> List[bool]:
    """recall↑ / latency_p50↓ 기준 파레토 여부 플래그."""
    flags = []
    for r in rows:
        dominated = any(
            (o["recall_at_k"] >= r["recall_at_k"] and
             o["latency_p50_ms"] <= r["latency_p50_ms"] and
             (o["recall_at_k"] > r["recall_at_k"] or
              o["latency_p50_ms"] < r["latency_p50_ms"]))
            for o in rows)
        flags.append(not dominated)
    return flags


def plot_results(rows: List[Dict], out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (1) recall vs latency — 파레토 강조
    fig, ax = plt.subplots(figsize=(7, 5))
    quants = sorted({r["quant"] for r in rows})
    cmap = {q: c for q, c in zip(quants, ["#4C78A8", "#F58518", "#54A24B"])}
    for r in rows:
        ax.scatter(r["latency_p50_ms"], r["recall_at_k"],
                   s=90 if r["pareto"] else 35,
                   edgecolor="black" if r["pareto"] else "none",
                   color=cmap[r["quant"]], alpha=0.85)
    par = sorted([r for r in rows if r["pareto"]], key=lambda x: x["latency_p50_ms"])
    if par:
        ax.plot([r["latency_p50_ms"] for r in par],
                [r["recall_at_k"] for r in par], "--", color="gray", lw=1)
    for q in quants:
        ax.scatter([], [], color=cmap[q], label=q)
    ax.set_xlabel("latency P50 (ms)"); ax.set_ylabel("recall@k")
    ax.set_title("HNSW sweep — recall vs latency (테두리=Pareto front)")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "pareto_recall_latency.png", dpi=150); plt.close(fig)

    # (2) quantization 비교 막대 (best nc 기준)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    best = {q: max((r for r in rows if r["quant"] == q),
                   key=lambda r: r["recall_at_k"]) for q in quants}
    for ax, key, label in zip(
            axes, ["recall_at_k", "latency_p50_ms", "store_mb"],
            ["recall@k ↑", "latency P50 ms ↓", "index store MB ↓"]):
        vals = [best[q].get(key) or 0 for q in quants]
        ax.bar(quants, vals, color=[cmap[q] for q in quants])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=9)
        ax.set_title(label)
    fig.suptitle("Quantization comparison (각 quant의 best-recall 설정)")
    fig.tight_layout()
    fig.savefig(out_dir / "quantization_comparison.png", dpi=150); plt.close(fig)

    # (3) ef_search(num_candidates) sweep 라인
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for q in quants:
        sub = sorted([r for r in rows if r["quant"] == q],
                     key=lambda r: r["num_candidates"])
        # 같은 nc가 여러 (m,efc)에서 나오므로 nc별 최고 recall만
        by_nc: Dict[int, Dict] = {}
        for r in sub:
            if r["num_candidates"] not in by_nc or \
               r["recall_at_k"] > by_nc[r["num_candidates"]]["recall_at_k"]:
                by_nc[r["num_candidates"]] = r
        xs = sorted(by_nc)
        ax.plot(xs, [by_nc[x]["recall_at_k"] for x in xs], "o-",
                color=cmap[q], label=f"{q} recall")
        ax2 = ax.twinx() if q == quants[0] else ax2
    ax.set_xlabel("num_candidates (ef_search)")
    ax.set_ylabel("recall@k"); ax.legend()
    ax.set_title("ef_search sweep — recall vs num_candidates")
    fig.tight_layout()
    fig.savefig(out_dir / "ef_search_sweep.png", dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-index", required=True)
    p.add_argument("--dims", type=int, default=64)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--m-list", type=int, nargs="+", default=[16, 32])
    p.add_argument("--efc-list", type=int, nargs="+", default=[100, 200])
    p.add_argument("--nc-list", type=int, nargs="+", default=[20, 50, 100, 200])
    p.add_argument("--quant-list", nargs="+", default=["float32", "int8_hnsw"],
                   choices=["float32", "int8_hnsw", "bbq_hnsw"])
    p.add_argument("--output-dir", default="./results/vector_db_sweep")
    p.add_argument("--skip-build", action="store_true",
                   help="변형 인덱스가 이미 있을 때 생성 생략")
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 쿼리 + ground-truth (exact KNN — 소스 float32 인덱스 기준으로 고정.
    # quantization 인덱스의 recall도 "float32 정답 이웃 대비"로 측정해야
    # 양자화로 인한 recall 손실이 드러난다.)
    print("[1] query 벡터 샘플링 + exact KNN ground-truth 계산 ...")
    queries = sample_query_vectors(args.source_index, args.n_queries)
    gt = [exact_topk(args.source_index, v, args.k) for v in queries]

    rows: List[Dict] = []
    combos = list(product(args.m_list, args.efc_list, args.quant_list))
    for m, efc, quant in combos:
        print(f"[2] variant m={m} efc={efc} quant={quant}")
        idx = (variant_name(args.source_index, m, efc, quant)
               if args.skip_build
               else create_variant(args.source_index, m, efc, quant, args.dims))
        mem = measure_memory(idx)
        for nc in args.nc_list:
            r = measure_config(idx, queries, gt, args.k, nc)
            rows.append({"index": idx, "m": m, "ef_construction": efc,
                         "quant": quant, "num_candidates": nc, **r, **mem})
            print(f"    nc={nc:4d}  recall={r['recall_at_k']:.4f}  "
                  f"p50={r['latency_p50_ms']}ms  p99={r['latency_p99_ms']}ms")

    flags = pareto_front(rows)
    for r, f in zip(rows, flags):
        r["pareto"] = f

    # CSV 저장
    cols = ["index", "m", "ef_construction", "quant", "num_candidates",
            "recall_at_k", "latency_p50_ms", "latency_p99_ms",
            "store_mb", "knn_vectors_mb", "pareto"]
    csv = "\n".join([",".join(cols)] +
                    [",".join(str(r.get(c, "")) for c in cols) for r in rows])
    (out_dir / "sweep_results.csv").write_text(csv)
    (out_dir / "sweep_results.json").write_text(json.dumps(rows, indent=2))

    plot_results(rows, out_dir)
    par = [r for r in rows if r["pareto"]]
    print(f"\n[done] {len(rows)} configs, Pareto {len(par)}개 → {out_dir}")
    for r in sorted(par, key=lambda x: x["latency_p50_ms"]):
        print(f"  ★ {r['quant']} m={r['m']} efc={r['ef_construction']} "
              f"nc={r['num_candidates']}: recall={r['recall_at_k']} "
              f"p50={r['latency_p50_ms']}ms store={r['store_mb']}MB")


if __name__ == "__main__":
    main()

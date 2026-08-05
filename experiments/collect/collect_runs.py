#!/usr/bin/env python3
"""
collect_runs.py — run 산출물 → tidy long 테이블 1개
=============================================================================
입력 (experiments/results/ 아래):
  run/<run_id>/manifest.json            축 값 + status(ok|suspect|failed)
  run/<run_id>/agent_eval_report.json   지표
  run/<run_id>/window.json              Prometheus 시간창
  index/<index_id>/index_stats.jsonl    store/working-set/segments
  index/<index_id>/manifest.json

출력:
  results/_rollup/runs_long.csv         tidy long (모든 지표 1행 1값)
  results/_rollup/runs_wide.csv         run × 주요지표 피벗 (사람이 읽는 용)
  results/_rollup/index_stats.csv       인덱스 메모리 축

설계:
  · 지표 추출은 "경로 → 컬럼명" 선언 테이블 하나로 끝낸다.
    report 스키마가 바뀌면 _METRIC_PATHS 만 고친다.
  · pandas 없이 stdlib 만으로 동작 (Windows/Cloud SDK python 3.9 호환).
  · 파일이 없거나 키가 없으면 조용히 건너뛴다 — 부분 실패한 매트릭스에서도
    끝까지 돈다.

사용:
  python experiments/collect/collect_runs.py
  python experiments/collect/collect_runs.py --results experiments/results
  python experiments/collect/collect_runs.py --include-suspect
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# 지표 선언 — (컬럼명, report 내 경로)
#   경로는 점(.) 구분. 리스트 인덱스는 지원하지 않는다(불필요).
# ============================================================
_METRIC_PATHS: List[tuple] = [
    # ── 검출 성능 ────────────────────────────────────────────
    ("distance_auroc",        "dist_auroc.distance_auroc"),
    ("agent_auroc",           "anomaly_auroc.auroc"),
    ("agent_nd_auroc",        "agent_nd_dual.agent_nd_auroc_all"),
    ("best_threshold",        "threshold_info.best_threshold"),
    ("dist_acc",              "distance_at_threshold.accuracy"),
    ("dist_fnr",              "distance_at_threshold.fnr"),
    ("dist_fpr",              "distance_at_threshold.fpr"),
    ("cls_nd_acc",            "agent_nd_dual.cls_at_nd_threshold.accuracy"),
    ("cls_nd_fnr",            "agent_nd_dual.cls_at_nd_threshold.fnr"),
    ("cls_nd_fpr",            "agent_nd_dual.cls_at_nd_threshold.fpr"),
    ("mean_normal_dist",      "dist_auroc.mean_normal_dist"),
    ("mean_anomaly_dist",     "dist_auroc.mean_anomaly_dist"),
    ("researched_mean_nd",    "agent_nd_dual.researched_mean_nd"),
    ("raw_mean_nd",           "agent_nd_dual.raw_mean_nd"),
    ("nd_recal_p50",          "agent_nd_dual.recalibrated.p50"),
    ("nd_recal_auroc",        "agent_nd_dual.recalibrated.auroc_nd_recal"),
    ("cls_recal_acc",         "agent_nd_dual.recalibrated.cls_at_recal_threshold.accuracy"),

    # ── 검색 품질 (CV C1/C2/C3) ──────────────────────────────
    ("routing_accuracy",      "router_retriever.routing_accuracy.cluster_assignment_accuracy"),
    ("router_mrr",            "router_retriever.router_mrr.router_mrr"),
    ("candidate_recall5",     "router_retriever.candidate_recall.candidate_recall_top5"),
    ("routing_eps_recall10",  "router_retriever.routing_epsilon_recall.routing_epsilon_recall_10pct"),
    ("coreset_eps_recall10",  "router_retriever.coreset_epsilon_recall.coreset_epsilon_recall_10pct"),
    ("recall5_raw",           "router_retriever.retrieval_recall.recall@5_raw"),
    ("recall5_adj",           "router_retriever.retrieval_recall.recall@5_coreset_adjusted"),

    # ── Agent 동작 ───────────────────────────────────────────
    ("expand_rate",           "action_distribution.expand_rate"),
    ("reroute_rate",          "action_distribution.reroute_rate"),
    ("cm_ratio",              "action_distribution.cm_ratio"),
    ("ni_ratio",              "action_distribution.ni_ratio"),
    ("verdict_change_rate",   "verdict_change.verdict_change_rate"),
    ("boundary_fnr",          "boundary.boundary_fnr"),
    ("boundary_expand_rate",  "boundary.boundary_expand_rate"),

    # ── 재검색 인과 (research_effect) ────────────────────────
    ("re_auroc_pre",          "research_effect.auroc.pre"),
    ("re_auroc_post",         "research_effect.auroc.post"),
    ("re_auroc_delta",        "research_effect.auroc.delta"),
    ("re_n_flip",             "research_effect.transitions.n_flip"),
    ("re_rescued",            "research_effect.transitions.rescued.n"),
    ("re_degraded",           "research_effect.transitions.degraded.n"),
    ("re_net_gain",           "research_effect.transitions.net_gain"),
    ("re_delta_nd_mean",      "research_effect.delta_nd.mean"),
    ("re_delta_nd_abs",       "research_effect.delta_nd.mean_abs"),
    ("re_dir_acc",            "research_effect.delta_nd.directional_accuracy"),
    ("re_n_researched",       "research_effect.delta_nd.n_researched"),
    ("re_amb_precision",      "research_effect.selection.ambiguous_precision"),
    ("re_amb_recall",         "research_effect.selection.ambiguous_recall"),
    ("re_missed",             "research_effect.selection.missed"),
    ("re_ms_mean",            "research_effect.cost.research_ms_mean"),
    ("re_ms_per_rescue",      "research_effect.cost.ms_per_rescue"),

    # ── Agent 축 (stage 3) ───────────────────────────────────
    ("mcp_tool_delta_mean",   "latency_breakdown.tool_delta_ms.mean"),
    ("mcp_tool_delta_p95",    "latency_breakdown.tool_delta_ms.p95"),
    ("ctrl_loop_mean",        "latency_breakdown.controller_loop_ms.mean"),
    ("ctrl_loop_p95",         "latency_breakdown.controller_loop_ms.p95"),
    ("agent_fnr",             "agent_at_threshold.fnr"),
    ("agent_fpr",             "agent_at_threshold.fpr"),
    ("agent_acc",             "agent_at_threshold.accuracy"),
    ("path_expand_n",         "action_distribution.expand_n"),
    ("path_accept_n",         "action_distribution.accept_direct_n"),

    # ── Latency ──────────────────────────────────────────────
    ("lat_total_mean",        "latency_breakdown.total_ms.mean"),
    ("lat_total_p50",         "latency_breakdown.total_ms.p50"),
    ("lat_total_p95",         "latency_breakdown.total_ms.p95"),
    ("lat_base_mean",         "latency_breakdown.base_retrieve_ms.mean"),
    ("lat_base_p95",          "latency_breakdown.base_retrieve_ms.p95"),
    ("lat_tool_mean",         "latency_breakdown.tool_delta_ms.mean"),
    ("lat_tool_p95",          "latency_breakdown.tool_delta_ms.p95"),
    ("ttft_p50",              "latency_breakdown.ttft_ms.p50"),
    ("ttft_p95",              "latency_breakdown.ttft_ms.p95"),
    ("tpot_p50",              "latency_breakdown.tpot_ms.p50"),

    # ── 메타 ─────────────────────────────────────────────────
    ("n_total",               "n_total"),
    ("n_error",               "n_error"),
    ("wall_total_s",          "wall_time.total_s"),
]

# 축 컬럼 (manifest.config 또는 report.axes 에서)
_AXIS_KEYS = [
    "VEC_QUANT", "HNSW_M", "HNSW_EF_CONSTRUCTION", "VEC_INDEX_CORESET_PCT",
    "RETRIEVER_OVERSAMPLE", "KNN_NUM_CANDIDATES", "SEARCH_MODE",
    "PERCOLATE_QUERY_VERSION", "ENABLE_BUCKETIZATION", "SKIP_ANALYZER",
    "TOOL_EXECUTION_MODE", "TOOL_GATE_ENABLE", "RESEARCH_MODE", "EXPERIMENT_CASE",
    "N_SHARDS", "PERCOLATOR_SHARD_DISTRIBUTION",
    "BUCKET_TOLERANCE", "BUCKET_MIN_MATCH",
]

# # _AXIS_KEYS 근처에 추가
# _CONTROLLER_ONLY_KEYS = {
#     "TOOL_GATE_ENABLE", "PATCHCORE_ON_RESEARCH", "PATCHCORE_TAU",
#     "PATCHCORE_REWEIGHT_MODE", "EVALUATOR_ONCE", "AGENT_MAX_ROUNDS",
#     "SKIP_ANALYZER_ON_CONFIDENT", "EXPAND_PLATEAU_TAU",
#     "REROUTE_ON_CENTROID", "REROUTE_SCORE_RATIO",
#     # ★ 양쪽에 중복 존재 — 반드시 동시 patch 해야 갈라지지 않는다
#     "CLS_ND_THRESHOLD", "SUFFICIENCY_BOUNDARY_LOW", "SUFFICIENCY_BOUNDARY_HIGH",
# }

# # run 렌더 블록에 1개 파일 추가
# _ctrl = {k: str(v) for k, v in r["config"].items() if k in _CONTROLLER_ONLY_KEYS}
# if _ctrl:
#     (d / "controller.patch.json").write_text(
#         json.dumps({"data": _ctrl}, indent=2, ensure_ascii=False), encoding="utf-8")

def _dig(obj: Any, path: str) -> Optional[Any]:
    """'a.b.c' 경로로 중첩 dict 탐색. 없으면 None."""
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _load_json(p: Path) -> Optional[dict]:
    try:
        return json.load(io.open(p, encoding="utf-8"))
    except Exception:
        return None


# ============================================================
# 1) run 수집
# ============================================================

def collect_runs(results: Path, include_suspect: bool) -> tuple:
    long_rows: List[Dict] = []
    wide_rows: List[Dict] = []
    skipped: List[str] = []

    run_dir = results / "run"
    if not run_dir.is_dir():
        return long_rows, wide_rows, skipped

    for d in sorted(run_dir.iterdir()):
        if not d.is_dir():
            continue
        run_id = d.name
        man = _load_json(d / "manifest.json") or {}
        rep = _load_json(d / "agent_eval_report.json")
        win = _load_json(d / "window.json") or {}

        status = man.get("status", "unknown")
        if rep is None:
            skipped.append(f"{run_id}(report 없음, status={status})")
            continue
        if status == "failed" or (status == "suspect" and not include_suspect):
            skipped.append(f"{run_id}(status={status})")
            continue

        cfg = dict(man.get("config") or {})
        cfg.update(rep.get("axes") or {})          # report.axes 가 더 신뢰됨
        base = {
            "run_id":     run_id,
            "group":      man.get("group", run_id.split("-")[0]),
            "variant":    man.get("variant", ""),
            "index_id":   man.get("index_id", ""),
            "stage":      man.get("stage", ""),
            "eval":       man.get("eval", ""),
            "status":     status,
            "n_samples":  man.get("n_samples", rep.get("n_total")),
            "started_at": win.get("from", ""),
            "finished_at": win.get("to", ""),
        }
        for k in _AXIS_KEYS:
            base[k.lower()] = cfg.get(k, "")

        wide = dict(base)
        for col, path in _METRIC_PATHS:
            v = _dig(rep, path)
            wide[col] = "" if v is None else v
            if v is not None and isinstance(v, (int, float)):
                long_rows.append({**base, "metric": col, "value": v})
        wide_rows.append(wide)

    return long_rows, wide_rows, skipped


# ============================================================
# 2) index 통계 수집
# ============================================================

def collect_locust(results: Path) -> List[Dict]:
    """[STEP6] locust/<run_id>/summary.tsv → 부하 단계별 행."""
    rows: List[Dict] = []
    d0 = results / "locust"
    if not d0.is_dir():
        return rows
    for d in sorted(d0.iterdir()):
        if not d.is_dir():
            continue
        f = d / "summary.tsv"
        if not f.exists():
            continue
        meta = _load_json(d / "locust_meta.json") or {}
        lines = io.open(f, encoding="utf-8").read().splitlines()
        if len(lines) < 2:
            continue
        hdr = lines[0].split("\t")
        for ln in lines[1:]:
            vals = ln.split("\t")
            if len(vals) != len(hdr):
                continue
            r = dict(zip(hdr, vals))
            r["run_id"] = d.name
            r["restarts_delta"] = (int(meta.get("restarts_after", 0))
                                   - int(meta.get("restarts_before", 0)))
            rows.append(r)
    return rows


def collect_indices(results: Path) -> List[Dict]:
    rows: List[Dict] = []
    idx_dir = results / "index"
    if not idx_dir.is_dir():
        return rows
    for d in sorted(idx_dir.iterdir()):
        if not d.is_dir():
            continue
        man = _load_json(d / "manifest.json") or {}
        cfg = man.get("config") or {}
        f = d / "index_stats.jsonl"
        if not f.exists():
            continue
        for line in io.open(f, encoding="utf-8").read().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            name = r.get("index", "")
            # 인덱스명에서 coreset % 추출: ..._percentage_{P}_cluster_tree_vec...
            pct = ""
            if "_percentage_" in name:
                pct = name.split("_percentage_")[1].split("_")[0]
            rows.append({
                "index_id":   d.name,
                "index":      name,
                "coreset_pct": pct,
                "vec_quant":  cfg.get("VEC_QUANT", ""),
                "hnsw_m":     cfg.get("HNSW_M", ""),
                "hnsw_ef":    cfg.get("HNSW_EF_CONSTRUCTION", ""),
                "docs":       r.get("docs", ""),
                "store_mb":   r.get("store_mb", ""),
                "working_set_mb": r.get("vector_working_set_mb", ""),
                "segments":   r.get("segments", ""),
            })
    return rows


# ============================================================
# 3) CSV 쓰기
# ============================================================

def _write_csv(path: Path, rows: List[Dict], cols: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        io.open(path, "w", encoding="utf-8").write("")
        print(f"  (빈 파일) {path}")
        return
    if cols is None:
        cols, seen = [], set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k); cols.append(k)
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  {path}  ({len(rows)} rows x {len(cols)} cols)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/results")
    ap.add_argument("--out", default="")
    ap.add_argument("--include-suspect", action="store_true",
                    help="pod 재시작 중 측정된 run 도 포함")
    args = ap.parse_args()

    results = Path(args.results)
    if not results.is_dir():
        sys.exit(f"결과 디렉터리 없음: {results}")
    out = Path(args.out) if args.out else results / "_rollup"

    long_rows, wide_rows, skipped = collect_runs(results, args.include_suspect)
    idx_rows = collect_indices(results)
    loc_rows = collect_locust(results)

    print(f"\n[collect] runs={len(wide_rows)}  metrics={len(long_rows)}  "
          f"indices={len(idx_rows)}  locust={len(loc_rows)}  skipped={len(skipped)}")
    for s in skipped:
        print(f"  skip: {s}")

    # ★ run 이 참조하는 index_id 중 통계가 없는 것을 경고한다.
    #   (--resume 으로 ingest 를 건너뛰었거나 ingest 로그 캡처가 실패한 경우)
    #   메모리 컬럼이 통째로 비면 T1-B/T1-C 표의 핵심 수치가 사라진다.
    have = {r["index_id"] for r in idx_rows}
    need = {r.get("index_id") for r in wide_rows if r.get("index_id")}
    missing = sorted(need - have)
    if missing:
        print(f"\n  ⚠ index_stats 없음: {', '.join(missing)}")
        print(f"    → 메모리 컬럼(store/working set)이 '—' 로 표시된다.")
        for m in missing:
            print(f"    복구: bash experiments/runner/run_case.sh ingest {m}")
    print()
    _write_csv(out / "runs_long.csv", long_rows)
    _write_csv(out / "runs_wide.csv", wide_rows)
    _write_csv(out / "index_stats.csv", idx_rows)
    _write_csv(out / "locust.csv", loc_rows)
    if not (out / "prom.csv").exists():
        print("  ⓘ prom.csv 없음 — page fault / MCP transport / TTFT 지표가 비어 있다.")
        print("    python experiments/collect/prom_snapshot.py --results "
              f"{results}")

    # 요약 JSON — sweep_report.py 가 바로 읽는다
    summary = {
        "n_runs": len(wide_rows),
        "n_indices": len(idx_rows),
        "skipped": skipped,
        "groups": sorted({r["group"] for r in wide_rows}),
        "run_ids": [r["run_id"] for r in wide_rows],
    }
    (out / "collect_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {out / 'collect_summary.json'}")


if __name__ == "__main__":
    main()

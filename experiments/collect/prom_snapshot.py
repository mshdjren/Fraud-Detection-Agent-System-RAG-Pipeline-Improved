#!/usr/bin/env python3
"""
prom_snapshot.py — run 시간창(window.json)으로 Prometheus 지표를 회수
=============================================================================
지금까지 계측만 되고 리포트에 한 번도 나오지 않던 지표들을 가져온다:

  · retriever_page_fault_delta          quantization/coreset ↔ 메모리 병목 연결
  · mcp_transport_latency_seconds{mode} 순수 transport overhead (CV C4)
      ↳ tool_delta 로 재면 analyzer 추론 수 초에 묻혀 12.5ms 를 못 잰다
  · analyzer_ttft_seconds / analyzer_tpot_ms   TTFT/TPOT (CV C8)
  · percolate_matching_time_ms          샤드 분산 효과 (T4-H)
  · retriever_es_index_store_bytes      인덱스 메모리 교차검증
  · agent_path_total / plan_action_total  경로 분포

접속 경로 (자동 탐지 순):
  1) --url 로 지정
  2) PROM_URL 환경변수
  3) http://localhost:19090   ← docker-compose 의 gmp-frontend
  4) http://localhost:9090

출력:
  results/<date>/run/<run_id>/prom_snapshot.json
  results/<date>/_rollup/prom.csv               (collect 가 읽는 tidy long)

사용:
  python experiments/collect/prom_snapshot.py --results experiments/results/0803
  python experiments/collect/prom_snapshot.py --results ... --only c-quant-f32
  python experiments/collect/prom_snapshot.py --results ... --url http://localhost:19090
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

# ============================================================
# 질의 정의 — (컬럼명, PromQL, 집계방식)
#   집계: "last"(구간 마지막 값) | "avg" | "max" | "sum"
#   {LABEL} 은 run_id 로 치환된다 (retriever/router 는 experiment_label 을 갖는다)
# ============================================================
_QUERIES: List[tuple] = [
    # ── 메모리 병목 ──────────────────────────────────────────
    ("page_fault_p50",
     'histogram_quantile(0.50, sum by (le) (rate(retriever_page_fault_delta_bucket[2m])))', "max"),
    ("page_fault_p95",
     'histogram_quantile(0.95, sum by (le) (rate(retriever_page_fault_delta_bucket[2m])))', "max"),
    ("page_fault_rate",
     'sum(rate(retriever_page_fault_delta_sum[2m]))', "avg"),
    ("es_jvm_heap_pct", 'avg(retriever_es_jvm_heap_percent)', "avg"),
    ("es_query_cache_mb", 'avg(retriever_es_query_cache_bytes) / 1048576', "avg"),

    # ── MCP transport (★ 순수 overhead — tool_delta 로는 못 잰다) ──
    ("mcp_p50_ms",
     'histogram_quantile(0.50, sum by (le) (rate(mcp_transport_latency_seconds_bucket[2m]))) * 1000', "max"),
    ("mcp_p95_ms",
     'histogram_quantile(0.95, sum by (le) (rate(mcp_transport_latency_seconds_bucket[2m]))) * 1000', "max"),
    ("mcp_calls", 'sum(increase(mcp_transport_latency_seconds_count[10m]))', "last"),
    ("mcp_errors",
     'sum(increase(mcp_transport_latency_seconds_count{status="error"}[10m]))', "last"),

    # ── LLM 추론 분해 ────────────────────────────────────────
    ("ttft_p50_s",
     'histogram_quantile(0.50, sum by (le) (rate(analyzer_ttft_seconds_bucket[2m])))', "max"),
    ("ttft_p95_s",
     'histogram_quantile(0.95, sum by (le) (rate(analyzer_ttft_seconds_bucket[2m])))', "max"),
    ("tpot_p50_ms",
     'histogram_quantile(0.50, sum by (le) (rate(analyzer_tpot_ms_bucket[2m])))', "max"),
    ("out_tokens_p50",
     'histogram_quantile(0.50, sum by (le) (rate(analyzer_output_tokens_bucket[2m])))', "max"),
    ("analyzer_calls", 'sum(increase(analyzer_requests_total[10m]))', "last"),
    ("analyzer_parse_err", 'sum(increase(analyzer_parse_errors_total[10m]))', "last"),

    # ── 검색 ────────────────────────────────────────────────
    ("retriever_p50_ms",
     'histogram_quantile(0.50, sum by (le) (rate(retriever_search_latency_ms_bucket[2m])))', "max"),
    ("retriever_p95_ms",
     'histogram_quantile(0.95, sum by (le) (rate(retriever_search_latency_ms_bucket[2m])))', "max"),
    ("percolate_p50_ms",
     'histogram_quantile(0.50, sum by (le) (rate(percolate_matching_time_ms_bucket[2m])))', "max"),
    ("percolate_p95_ms",
     'histogram_quantile(0.95, sum by (le) (rate(percolate_matching_time_ms_bucket[2m])))', "max"),
    ("top1_dist_p50",
     'histogram_quantile(0.50, sum by (le) (rate(retriever_top1_distance_bucket[2m])))', "max"),

    # ── agent 경로 ──────────────────────────────────────────
    ("agent_rounds_p95",
     'histogram_quantile(0.95, sum by (le) (rate(agent_rounds_per_request_bucket[2m])))', "max"),
    ("planner_fail_rate",
     'sum(rate(planner_calls_total{status!="success"}[5m])) / '
     'clamp_min(sum(rate(planner_calls_total[5m])), 0.001)', "avg"),
]


# ============================================================
# Prometheus HTTP API
# ============================================================

def _http_json(url: str, timeout: int = 20) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def detect_url(explicit: str = "") -> Optional[str]:
    import os
    cands = [c for c in (explicit, os.getenv("PROM_URL", ""),
                         "http://localhost:19090", "http://localhost:9090") if c]
    for u in cands:
        u = u.rstrip("/")
        if _http_json(f"{u}/api/v1/query?query=up") is not None:
            return u
    return None


def query_range(base: str, expr: str, start: str, end: str, step: str = "30s") -> List[float]:
    """구간 질의 → 숫자 배열. 실패/빈 결과면 []."""
    q = urllib.parse.urlencode({"query": expr, "start": start, "end": end, "step": step})
    d = _http_json(f"{base}/api/v1/query_range?{q}")
    if not d or d.get("status") != "success":
        return []
    vals: List[float] = []
    for series in (d.get("data", {}).get("result") or []):
        for _ts, v in series.get("values", []):
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if f == f and abs(f) != float("inf"):      # NaN/Inf 제외
                vals.append(f)
    return vals


def _agg(vals: List[float], how: str) -> Optional[float]:
    if not vals:
        return None
    if how == "last":
        return round(vals[-1], 4)
    if how == "max":
        return round(max(vals), 4)
    if how == "sum":
        return round(sum(vals), 4)
    return round(sum(vals) / len(vals), 4)


# ============================================================
# run 단위 스냅샷
# ============================================================

def snapshot_run(base: str, run_dir: Path, pad_s: int = 30) -> Optional[Dict]:
    wf = run_dir / "window.json"
    if not wf.exists():
        return None
    try:
        w = json.load(io.open(wf, encoding="utf-8"))
        start, end = w["from"], w["to"]
    except Exception:
        return None

    # ★ 시간창을 앞뒤로 조금 넓힌다. rate()/histogram_quantile 은 lookback 이
    #   필요해서 정확히 창 경계만 주면 값이 비어 나온다.
    import datetime as dt

    def _shift(iso: str, sec: int) -> str:
        t = dt.datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
        return (t + dt.timedelta(seconds=sec)).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        s2, e2 = _shift(start, -pad_s), _shift(end, pad_s)
    except Exception:
        s2, e2 = start, end

    out: Dict = {"run_id": run_dir.name, "window": {"from": s2, "to": e2}}
    n_ok = 0
    for col, expr, how in _QUERIES:
        v = _agg(query_range(base, expr, s2, e2), how)
        out[col] = v
        if v is not None:
            n_ok += 1
    out["_n_metrics_ok"] = n_ok
    out["_n_metrics_total"] = len(_QUERIES)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/results",
                    help="날짜 폴더까지 지정 (예: experiments/results/0803)")
    ap.add_argument("--url", default="", help="Prometheus/GMP frontend URL")
    ap.add_argument("--only", default="", help="쉼표구분 run_id 필터")
    ap.add_argument("--step", default="30s")
    args = ap.parse_args()

    results = Path(args.results)
    # 날짜 폴더가 아니라 results/ 를 받았으면 가장 최근 날짜를 고른다
    if not (results / "run").is_dir():
        cands = sorted([d for d in results.iterdir()
                        if d.is_dir() and d.name.isdigit() and (d / "run").is_dir()])
        if cands:
            results = cands[-1]
            print(f"[prom] 날짜 폴더 자동 선택: {results}")
    if not (results / "run").is_dir():
        sys.exit(f"run 디렉터리 없음: {results}/run")

    base = detect_url(args.url)
    if not base:
        print("❌ Prometheus 접속 불가. 아래 중 하나로 띄운 뒤 재시도:")
        print("   docker compose up -d gmp-frontend      # → http://localhost:19090")
        print("   kubectl port-forward -n gmp-system svc/frontend 19090:9090")
        print("   또는 --url http://... 로 직접 지정")
        sys.exit(1)
    print(f"[prom] {base}")

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    rows: List[Dict] = []
    for d in sorted((results / "run").iterdir()):
        if not d.is_dir() or (only and d.name not in only):
            continue
        snap = snapshot_run(base, d, )
        if snap is None:
            print(f"  skip {d.name} (window.json 없음)")
            continue
        (d / "prom_snapshot.json").write_text(
            json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append(snap)
        print(f"  {d.name:<26} {snap['_n_metrics_ok']:>2}/{snap['_n_metrics_total']} metrics"
              f"   mcp_p95={snap.get('mcp_p95_ms')}ms"
              f"   pf_p95={snap.get('page_fault_p95')}"
              f"   ttft_p50={snap.get('ttft_p50_s')}s")

    if not rows:
        print("\n수집된 run 없음")
        return

    out = results / "_rollup"
    out.mkdir(parents=True, exist_ok=True)
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols and k != "window":
                cols.append(k)
    with io.open(out / "prom.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n  {out / 'prom.csv'}  ({len(rows)} rows x {len(cols)} cols)")

    # 전부 None 인 지표는 계측/스크랩 문제이므로 크게 알린다
    dead = [c for c, _, _ in _QUERIES if all(r.get(c) is None for r in rows)]
    if dead:
        print(f"\n  ⚠ 전 run 에서 값이 없는 지표 ({len(dead)}): {', '.join(dead)}")
        print("    → PodMonitoring 스크랩 여부 또는 메트릭 이름 확인:")
        print("      kubectl get podmonitoring -n elastic")
        print("      kubectl logs -n gmp-system -l app.kubernetes.io/name=collector --tail=50")


if __name__ == "__main__":
    main()

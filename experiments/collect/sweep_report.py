#!/usr/bin/env python3
"""
sweep_report.py — runs_wide.csv → 포트폴리오용 표 + 그래프
=============================================================================
Tier 순서로 산출한다. 이력서(CV) 주장 재현이 최우선.

  T1-A  percolate   : v1 / v14 / v14+bucket   → Router Recall@5, P99   (CV C1)
  T1-B  coreset     : 100 / 10 / 1 %          → 메모리, AUROC          (CV C2)
  T1-C  quant       : f32/int8os0/int8os3/bbq → 메모리, Recall, nd     (CV C3)
  T1-D  research    : 재검색 인과 분해                                   (신규 기여)
  T1-E  pareto      : x=P95, y=AUROC, size=working set                  (종합)

출력:
  _rollup/tables/*.md      README 에 그대로 붙일 markdown 표
  _rollup/plots/*.png      matplotlib (없으면 표만 생성하고 계속)
  _rollup/results.md       전체 묶음

pandas 불필요 — stdlib + (선택) matplotlib.

사용:
  python experiments/collect/sweep_report.py
  python experiments/collect/sweep_report.py --rollup experiments/results/_rollup
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Dict, List, Optional

# ── matplotlib 로드 ──────────────────────────────────────────────────
#  ★ NumPy 2.x 환경에서 NumPy 1.x 로 빌드된 matplotlib 을 import 하면
#    C 확장이 "_ARRAY_API not found" 로 깨지고, NumPy 가 stderr 에 직접
#    긴 경고를 쏟는다(파이썬 예외 처리로는 안 가려짐). import 구간만
#    stderr 를 죽여서 조용히 처리하고, 실패 시 조치법을 한 줄로 알린다.
MPL_OK = False
MPL_ERR = ""


def _load_mpl():
    global MPL_OK, MPL_ERR, plt
    import contextlib
    import os as _os
    import sys as _sys
    buf = io.StringIO()
    devnull = open(_os.devnull, "w")
    saved = _sys.stderr
    try:
        _sys.stderr = devnull                      # C 레벨 경고 억제
        with contextlib.redirect_stderr(buf):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            import warnings
            warnings.filterwarnings("ignore", message="Glyph .* missing from font")
            for _f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
                try:
                    from matplotlib import font_manager
                    if any(_f == f.name for f in font_manager.fontManager.ttflist):
                        _plt.rcParams["font.family"] = _f
                        _plt.rcParams["axes.unicode_minus"] = False
                        break
                except Exception:
                    pass
        plt = _plt
        MPL_OK = True
    except BaseException as e:                     # AttributeError/ImportError 모두
        MPL_ERR = f"{type(e).__name__}: {e}"
    finally:
        _sys.stderr = saved
        devnull.close()


_load_mpl()


def _mpl_hint() -> str:
    if MPL_OK:
        return ""
    hint = "  (matplotlib 사용 불가 — 표만 생성)"
    if "_ARRAY_API" in MPL_ERR or "NumPy" in MPL_ERR or "numpy" in MPL_ERR:
        hint += ("\n  원인: matplotlib 이 NumPy 1.x 로 빌드됐는데 런타임은 NumPy 2.x"
                 "\n  해결(둘 중 하나):"
                 "\n    python -m pip install -U \"matplotlib>=3.9\""
                 "\n    python -m pip install \"numpy<2\"")
    elif MPL_ERR:
        hint += f"\n  원인: {MPL_ERR}"
        hint += "\n  해결: python -m pip install -U matplotlib"
    return hint


# ============================================================
# 유틸
# ============================================================

def read_csv(p: Path) -> List[Dict]:
    if not p.exists() or p.stat().st_size == 0:
        return []
    with io.open(p, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def num(v, nd: int = 4) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, nd)


def fmt(v, nd: int = 4, pct: bool = False) -> str:
    f = num(v, nd)
    if f is None:
        return "—"
    return f"{f * 100:.1f}%" if pct else f"{f:g}"


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_(데이터 없음)_\n"
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def pick(runs: List[Dict], group_prefix: str) -> List[Dict]:
    """run_id 접두로 그룹 선택 (group 컬럼이 비어도 동작)."""
    sel = [r for r in runs if r.get("run_id", "").startswith(group_prefix)]
    return sorted(sel, key=lambda r: r["run_id"])


def index_map(idx: List[Dict]) -> Dict:
    """(index_id, coreset_pct) → index_stats 행.

    ★ vec_quant 나 (m, ef) 로 조인하면 안 된다. f32/m32ef200/m8ef64 는 모두
      vec_quant=f32 이고, f32/int8/bbq 는 모두 m=16,ef=100 이라 키가 충돌해
      마지막 행이 이긴다(실제로 T2-D 가 int8 의 690MB 를 표시했다).
      index_id 는 ingest Job 단위라 유일하다.
    """
    return {(i.get("index_id"), str(i.get("coreset_pct"))): i for i in idx}


def savefig(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {p}")


# ============================================================
# T1-A  Percolate  (CV C1)
# ============================================================

def table_percolate(runs: List[Dict]) -> str:
    sel = pick(runs, "a-percolate")
    rows = []
    for r in sel:
        rows.append([
            r["run_id"].replace("a-percolate-", ""),
            r.get("percolate_query_version", "—"),
            r.get("enable_bucketization", "—"),
            fmt(r.get("candidate_recall5")),
            fmt(r.get("routing_accuracy")),
            fmt(r.get("router_mrr")),
            fmt(r.get("recall5_raw")),
            fmt(r.get("lat_base_p95"), 1),
            fmt(r.get("lat_total_p95"), 1),
            fmt(r.get("distance_auroc")),
        ])
    body = md_table(
        ["variant", "version", "bucket", "cand_recall@5", "routing_acc",
         "MRR", "recall@5", "base P95(ms)", "total P95(ms)", "dist AUROC"], rows)
    note = ""
    if len(sel) >= 2:
        base = sel[0]
        deltas = []
        for r in sel[1:]:
            a, b = num(base.get("candidate_recall5")), num(r.get("candidate_recall5"))
            c, d = num(base.get("lat_base_p95")), num(r.get("lat_base_p95"))
            if a is not None and b is not None:
                deltas.append(f"`{r['run_id']}` recall@5 {(b - a) * 100:+.1f}%p")
            if c and d:
                deltas.append(f"P95 {(d - c) / c * 100:+.1f}%")
        if deltas:
            note = f"\n**vs `{base['run_id']}`** — " + " · ".join(deltas) + "\n"
    return body + note


# ============================================================
# T1-B  Coreset  (CV C2)
# ============================================================

def table_coreset(runs: List[Dict], idx: List[Dict]) -> str:
    sel = pick(runs, "b-coreset")
    # 인덱스 통계에서 coreset % 별 메모리
    imap = index_map(idx)
    rows = []
    for r in sel:
        pct = str(r.get("vec_index_coreset_pct", "")).strip()
        m = imap.get((r.get("index_id"), pct), {})
        rows.append([
            pct + "%",
            m.get("docs", "—"),
            fmt(m.get("store_mb"), 1),
            fmt(m.get("working_set_mb"), 1),
            fmt(r.get("distance_auroc")),
            fmt(r.get("recall5_adj")),      # ★ coreset 비교는 adjusted 가 정답
            fmt(r.get("recall5_raw")),
            fmt(r.get("coreset_eps_recall10")),
            fmt(r.get("lat_base_p50"), 1) if r.get("lat_base_p50") else fmt(r.get("lat_base_mean"), 1),
            fmt(r.get("lat_base_p95"), 1),
        ])
    body = md_table(
        ["coreset", "docs", "store(MB)", "working set(MB)", "dist AUROC",
         "**recall@5 adj**", "recall@5 raw", "ε-recall",
         "base mean(ms)", "base P95(ms)"], rows)
    note = ""
    if len(sel) >= 2:
        b100 = next((r for r in sel if str(r.get("vec_index_coreset_pct")) == "100"), None)
        b10 = next((r for r in sel if str(r.get("vec_index_coreset_pct")) == "10"), None)
        if b100 and b10:
            a, b = num(b100.get("distance_auroc")), num(b10.get("distance_auroc"))
            m1 = num(imap.get((b100.get("index_id"), "100"), {}).get("working_set_mb"))
            m2 = num(imap.get((b10.get("index_id"), "10"), {}).get("working_set_mb"))
            parts = []
            if m1 and m2:
                parts.append(f"메모리 {(m2 - m1) / m1 * 100:+.1f}%")
            if a is not None and b is not None:
                parts.append(f"AUROC {(b - a) * 100:+.2f}%p")
            note = "\n**10% vs 100%** — " + " · ".join(parts) + "\n"
    note += """
> **`recall@5 adj` 를 봐야 한다.** GT top-5 는 100% 인덱스 기준이라, 10% coreset
> 에서는 진짜 top-5 가 인덱스에 존재하지 않는다. `raw` 가 떨어지는 것은 성능
> 저하가 아니라 **구조적 필연**이며, coreset 커버리지로 보정한 `adj` 만 비교 가능하다.
>
> **latency 는 이 규모에서 의미 없다.** 100% 인덱스(≈285MB working set)가 docs 노드
> page cache(≈2Gi)에 전량 상주하므로 압축해도 page fault 가 발생하지 않는다.
> coreset 의 latency 이득은 **working set 이 page cache 를 초과할 때만** 발현된다.
"""
    return body + note


# ============================================================
# T1-C  Quantization  (CV C3)  ★ 핵심
# ============================================================

def table_quant(runs: List[Dict], idx: List[Dict]) -> str:
    sel = pick(runs, "c-quant")
    imap = index_map(idx)
    rows = []
    for r in sel:
        q = r.get("vec_quant", "")
        m = imap.get((r.get("index_id"), "100"), {})
        rows.append([
            r["run_id"].replace("c-quant-", ""),
            q or "—",
            fmt(r.get("retriever_oversample"), 1),
            fmt(m.get("store_mb"), 1),
            fmt(m.get("working_set_mb"), 1),
            fmt(r.get("distance_auroc")),
            fmt(r.get("recall5_raw")),
            fmt(r.get("best_threshold"), 3),
            fmt(r.get("cls_nd_acc")),
            fmt(r.get("cls_nd_fpr")),
            fmt(r.get("raw_mean_nd"), 3),
        ])
    body = md_table(
        ["variant", "quant", "oversample", "store(MB)", "working set(MB)",
         "**dist AUROC**", "recall@5", "best_thr", "cls@0.42 acc",
         "cls@0.42 FPR", "mean nd"], rows)
    return body + """
> **읽는 법**: `dist AUROC`(순위)와 `cls@0.42`(절대 임계)를 **반드시 분리해서** 본다.
> 양자화 잔차는 실제 최근접 거리와 맞먹는 크기로 **가법 상수처럼** 더해지므로
> (d_est² ≈ d_true² + ‖r‖²), 단조 변환이라 순위는 보존되고 스케일만 이동한다.
> → `int8os0`(rescore 없음)에서 **AUROC 는 유지되는데 `best_thr`/`cls@0.42` 만 무너지면**
> "거리값이 신호인 시스템에서 rescore 는 optional 이 아니라 필수"가 수치로 증명된다.
"""


# ============================================================
# T1-D  Research Effect
# ============================================================

def table_research(runs: List[Dict]) -> str:
    sel = [r for r in runs if num(r.get("re_n_researched"))]
    rows = []
    for r in sorted(sel, key=lambda x: x["run_id"]):
        rows.append([
            r["run_id"],
            r.get("re_n_researched", "—"),
            fmt(r.get("re_auroc_pre")),
            fmt(r.get("re_auroc_post")),
            fmt(r.get("re_auroc_delta")),
            r.get("re_rescued", "—"),
            r.get("re_degraded", "—"),
            r.get("re_net_gain", "—"),
            fmt(r.get("re_delta_nd_mean"), 3),
            fmt(r.get("re_dir_acc")),
            fmt(r.get("re_amb_precision")),
            fmt(r.get("re_ms_mean"), 1),
        ])
    body = md_table(
        ["run", "n_res", "AUROC pre", "post", "Δ", "rescued", "degraded",
         "**net_gain**", "Δnd mean", "**dir_acc**", "amb_prec", "ms/call"], rows)
    return body + """
> `net_gain = rescued − degraded`. verdict 가 하나도 안 바뀌어도
> `dir_acc`(Δnd 부호가 정답 방향인 비율)는 살아 있으므로,
> **"재검색 효과 없음"과 "효과를 측정 못 함"을 구분**할 수 있다.
> 0.5 근처면 재검색이 무작위, 0.8 이상이면 판정을 못 바꿨을 뿐 정보는 더한 것.
"""


# ============================================================
# T2-D  HNSW m / ef_construction  (CV C7)
# ============================================================

def table_hnsw(runs: List[Dict], idx: List[Dict]) -> str:
    sel = pick(runs, "d-hnsw")
    imap = index_map(idx)
    rows = []
    for r in sel:
        m = imap.get((r.get("index_id"), "100"), {})
        rows.append([
            r["run_id"].replace("d-hnsw-", ""),
            r.get("hnsw_m", "—"), r.get("hnsw_ef_construction", "—"),
            fmt(m.get("store_mb"), 1), fmt(m.get("working_set_mb"), 1),
            m.get("segments", "—"),
            fmt(r.get("distance_auroc")), fmt(r.get("recall5_raw")),
            fmt(r.get("candidate_recall5")),
            fmt(r.get("lat_base_mean"), 1), fmt(r.get("lat_base_p95"), 1),
        ])
    return md_table(
        ["variant", "m", "ef_c", "store(MB)", "working set(MB)", "segs",
         "dist AUROC", "recall@5", "cand_recall@5",
         "base mean(ms)", "base P95(ms)"], rows) + """
> `m` 은 그래프 연결도(메모리·recall 동시 상승), `ef_construction` 은 build 품질.
> **여기서 고른 운영점을 stage 3 전체에 고정**한다 — 검색이 흔들리면
> MCP/toolgate 측정이 교란된다.
"""


# ============================================================
# T2-E  num_candidates — recall vs latency 곡선
# ============================================================

def table_numcand(runs: List[Dict]) -> str:
    sel = sorted(pick(runs, "e-numcand"),
                 key=lambda r: num(r.get("knn_num_candidates")) or 0)
    rows = []
    for r in sel:
        rows.append([
            r.get("knn_num_candidates", "—"),
            fmt(r.get("distance_auroc")), fmt(r.get("recall5_raw")),
            fmt(r.get("candidate_recall5")),
            fmt(r.get("lat_base_mean"), 1), fmt(r.get("lat_base_p95"), 1),
            fmt(r.get("lat_total_p95"), 1),
        ])
    return md_table(
        ["num_candidates", "dist AUROC", "recall@5", "cand_recall@5",
         "base mean(ms)", "base P95(ms)", "total P95(ms)"], rows) + """
> **재적재 0** 인 유일한 검색 축 — 쿼리 파라미터라 ConfigMap patch 만으로 바뀐다.
> recall 이 평탄해지는 지점(knee)이 운영점이며, 그 이상은 latency 만 낸다.
"""


# ============================================================
# T2-I  bucket prefilter 튜닝 — routing_acc 붕괴 규명
# ============================================================

def table_bucket(runs: List[Dict]) -> str:
    sel = sorted(pick(runs, "i-bucket"), key=lambda r: r["run_id"])
    rows = []
    for r in sel:
        rows.append([
            r["run_id"].replace("i-bucket-", ""),
            r.get("bucket_tolerance", "—"), r.get("bucket_min_match", "—"),
            fmt(r.get("routing_accuracy")), fmt(r.get("router_mrr")),
            fmt(r.get("candidate_recall5")), fmt(r.get("recall5_raw")),
            fmt(r.get("lat_base_mean"), 1), fmt(r.get("lat_base_p95"), 1),
            fmt(r.get("distance_auroc")),
        ])
    return md_table(
        ["variant", "tolerance", "min_match", "**routing_acc**", "MRR",
         "cand_recall@5", "recall@5", "base mean(ms)", "base P95(ms)", "dist AUROC"],
        rows) + """
> v14bucket 에서 `routing_acc` 0.80 → **0.25**, MRR 1.00 → 0.57 로 무너졌다.
> 그런데 `cand_recall@5` 는 1.0 유지 — **정답 클러스터는 top-5 에 계속 있는데
> top-1 서열만 흔들린다.** prefilter 가 후보를 거르는 게 아니라 percolate
> 점수 서열을 희석시킨다는 뜻이다.
> feature 9개 중 `min_match` 개만 ±`tolerance` 버킷 안에 들면 통과하므로,
> 느슨할수록 살아남는 룰이 많아진다. `t0m7`(가장 엄격)에서 routing_acc 가
> 회복되면 원인이 확정된다. `base P95` 도 함께 떨어져야 정상
> (prefilter 가 매칭 대상을 줄였다면 빨라져야 하는데 v14bucket 은 오히려 늘었다).
"""


# ============================================================
# T3-F  MCP transport 3-axis  (CV C4)
# ============================================================

def table_mcp(runs: List[Dict]) -> str:
    order = {"direct": 0, "mcp": 1, "mcpnative": 2}
    sel = sorted(pick(runs, "f-mcp"),
                 key=lambda r: order.get(r["run_id"].replace("f-mcp-", ""), 9))
    rows, base = [], None
    for r in sel:
        v = r["run_id"].replace("f-mcp-", "")
        mean = num(r.get("mcp_tool_delta_mean"))
        if v == "direct":
            base = mean
        delta = f"{mean - base:+.1f}" if (mean is not None and base is not None) else "—"
        rows.append([
            v, r.get("tool_execution_mode", "—"),
            fmt(mean, 1), fmt(r.get("mcp_tool_delta_p95"), 1),
            fmt(r.get("ctrl_loop_mean"), 1), fmt(r.get("ctrl_loop_p95"), 1),
            delta, fmt(r.get("distance_auroc")),
        ])
    return md_table(
        ["transport", "mode", "tool Δ mean(ms)", "tool Δ P95(ms)",
         "ctrl loop mean", "ctrl loop P95", "**vs direct(ms)**", "dist AUROC"],
        rows) + """
> `direct` 를 기준선으로 한 **순수 transport overhead**. 세 경로 모두 동일
> wall-clock 구간(`mcp_transport_latency_seconds`)으로 감싸 측정된다.
> AUROC 가 세 행에서 동일해야 정상 — 다르면 transport 가 결과를 바꾼 것이므로 버그다.
"""


# ============================================================
# T3-G  conditional re-retrieval on/off  (CV C6)
# ============================================================

def table_toolgate(runs: List[Dict]) -> str:
    sel = sorted(pick(runs, "g-toolgate"), reverse=True,
                 key=lambda r: r["run_id"])
    rows = []
    for r in sel:
        rows.append([
            r["run_id"].replace("g-toolgate-", ""),
            r.get("tool_gate_enable", "—"),
            fmt(r.get("expand_rate")), r.get("path_expand_n", "—"),
            fmt(r.get("distance_auroc")), fmt(r.get("cls_nd_fnr")),
            fmt(r.get("cls_nd_fpr")), fmt(r.get("boundary_fnr")),
            fmt(r.get("lat_total_p95"), 1), fmt(r.get("re_net_gain")),
        ])
    return md_table(
        ["variant", "gate", "expand_rate", "expand_n", "dist AUROC",
         "FNR", "FPR", "boundary FNR", "total P95(ms)", "net_gain"], rows) + """
> CV C6 은 "FNR −4.5%p, P99 증가 +8~12% 억제" 를 주장한다.
> **off 대비 on 의 FNR 감소와 P95 증가를 같이 봐야** trade-off 가 성립한다.
"""


# ============================================================
# 재검색 진단 (Δnd=0 원인 규명)
# ============================================================

def table_research_diag(runs: List[Dict]) -> str:
    rows = []
    for r in sorted(runs, key=lambda x: x["run_id"]):
        n = num(r.get("re_n_researched"))
        if not n:
            continue
        rows.append([
            r["run_id"], int(n),
            fmt(r.get("re_delta_nd_abs"), 4),
            fmt(r.get("re_dir_acc")),
            r.get("re_n_flip", "—"),
            fmt(r.get("re_ms_mean"), 1),
        ])
    return md_table(["run", "n_res", "mean|Δnd|", "dir_acc", "n_flip", "ms/call"],
                    rows) + """
> `mean|Δnd| == 0` 이고 `dir_acc` 가 `—` 면 **재검색 nd 가 갱신되지 않은 것**이다
> (효과 없음이 아니라 측정 불가). 원인은 `agent_predictions.json` 의
> `agent_rounds[].patchcore_status` 로 구분된다:
> `tool_failed`(예외 삼킴) / `insufficient_neighbors`(이웃<2) / `off`(env) / `ok`(실제 무변화).
"""


# ============================================================
# T4-H  percolator 샤드 분산
# ============================================================

def table_percshard(runs: List[Dict]) -> str:
    sel = sorted(pick(runs, "h-percshard"), key=lambda r: num(r.get("n_shards")) or 0)
    rows, base = [], None
    for r in sel:
        p95 = num(r.get("lat_base_p95"))
        if base is None:
            base = p95
        gain = f"{(base - p95) / base * 100:+.1f}%" if (p95 and base) else "—"
        rows.append([
            r["run_id"].replace("h-percshard-", ""),
            r.get("n_shards", "—"),
            fmt(r.get("lat_base_mean"), 1), fmt(p95, 1), gain,
            fmt(r.get("candidate_recall5")), fmt(r.get("routing_accuracy")),
            fmt(r.get("distance_auroc")),
        ])
    return md_table(
        ["variant", "shards", "base mean(ms)", "base P95(ms)", "**vs s1**",
         "cand_recall@5", "routing_acc", "dist AUROC"], rows) + """
> percolate 는 등록 쿼리(5,725개)를 입력 문서에 MemoryIndex 로 실행하므로
> 비용이 **샤드당 쿼리 수**에 비례한다. 일반 역색인과 달리 CPU 작업이 실제로
> 나뉘어 near-linear 로 스케일한다 — **단 노드 CPU 코어 수까지만**
> (percolator 노드 limit 1800m ≈ 1.8코어 → 2배 부근에서 포화 예상).
> `recall`/`AUROC` 는 세 행에서 동일해야 정상 — 샤딩은 결과를 바꾸지 않는다.
"""


# ============================================================
# T6  locust 부하 3단계
# ============================================================

def table_locust(loc: List[Dict]) -> str:
    if not loc:
        return "_(locust 결과 없음 — run_locust.sh 미실행)_\n"
    rows = []
    for r in sorted(loc, key=lambda x: (x.get("run_id", ""), num(x.get("users")) or 0)):
        rows.append([
            r.get("run_id", "—"), r.get("level", "—"), r.get("users", "—"),
            r.get("rps", "—"), r.get("p50", "—"), r.get("p95", "—"), r.get("p99", "—"),
            r.get("fail", "—"), r.get("n_req", "—"),
            "⚠" if str(r.get("restarts_delta", "0")) not in ("0", "") else "",
        ])
    return md_table(
        ["run", "level", "users", "RPS", "P50", "P95", "P99", "fail", "n_req", "restart"],
        rows) + """
> `light`(5) → `medium`(20) → `heavy`(50). P99 가 users 에 **선형 이상으로**
> 증가하기 시작하는 지점이 포화점이다. `restart ⚠` 는 부하 중 pod 재시작이
> 있었다는 뜻으로, 실패가 아니라 **포화의 직접 증거**로 읽는다.
"""


# ============================================================
# P1  Prometheus — 메모리 병목 (quant/coreset ↔ page fault)
# ============================================================

def table_prom_memory(prom: List[Dict], runs: List[Dict], idx: List[Dict]) -> str:
    if not prom:
        return "_(prom.csv 없음 — prom_snapshot.py 미실행)_\n"
    imap = index_map(idx)
    rmap = {r["run_id"]: r for r in runs}
    rows = []
    for pr in sorted(prom, key=lambda x: x.get("run_id", "")):
        r = rmap.get(pr.get("run_id"), {})
        m = imap.get((r.get("index_id"), str(r.get("vec_index_coreset_pct"))), {})
        rows.append([
            pr.get("run_id", "—"), r.get("vec_quant", "—"),
            fmt(m.get("working_set_mb"), 1),
            fmt(pr.get("page_fault_p50"), 1), fmt(pr.get("page_fault_p95"), 1),
            fmt(pr.get("es_jvm_heap_pct"), 1),
            fmt(pr.get("retriever_p50_ms"), 1), fmt(pr.get("retriever_p95_ms"), 1),
            fmt(pr.get("top1_dist_p50"), 3),
        ])
    return md_table(
        ["run", "quant", "working set(MB)", "pf p50", "pf p95", "heap%",
         "retriever p50", "retriever p95", "top1 dist p50"], rows) + """
> **압축의 latency 이득은 page fault 가 날 때만 발현된다.** working set 이
> docs 노드 page cache 안에 전부 들어가면 f32 도 pf=0 이라 bbq 와 차이가 없다.
> `pf p95` 가 전 행 0 이면 실험 조건이 성립하지 않은 것이다 —
> docs 노드 `resources.limits.memory` 를 낮춰 page cache 를 줄이거나
> aug 인덱스로 데이터를 키워야 한다.
"""


# ============================================================
# P2  Prometheus — MCP 순수 transport (CV C4) · TTFT/TPOT (CV C8)
# ============================================================

def table_prom_llm(prom: List[Dict], runs: List[Dict]) -> str:
    if not prom:
        return "_(prom.csv 없음)_\n"
    rmap = {r["run_id"]: r for r in runs}
    rows, base = [], None
    for pr in sorted(prom, key=lambda x: x.get("run_id", "")):
        r = rmap.get(pr.get("run_id"), {})
        mode = r.get("tool_execution_mode", "—")
        p95 = num(pr.get("mcp_p95_ms"))
        if mode == "direct" and base is None:
            base = p95
        delta = f"{p95 - base:+.1f}" if (p95 is not None and base is not None) else "—"
        rows.append([
            pr.get("run_id", "—"), mode,
            fmt(pr.get("mcp_p50_ms"), 1), fmt(p95, 1), delta,
            fmt(pr.get("mcp_errors"), 0),
            fmt(pr.get("ttft_p50_s"), 3), fmt(pr.get("ttft_p95_s"), 3),
            fmt(pr.get("tpot_p50_ms"), 1), fmt(pr.get("out_tokens_p50"), 0),
        ])
    return md_table(
        ["run", "mode", "mcp p50(ms)", "**mcp p95(ms)**", "vs direct",
         "mcp err", "TTFT p50(s)", "TTFT p95(s)", "TPOT p50(ms)", "out tok"],
        rows) + """
> **`mcp p95` 가 CV C4 의 진짜 측정값이다.** T3-F 의 `tool Δ` 는
> `total − base_retrieve` 라 analyzer 추론 수 초가 통째로 섞여 있어
> 수십 ms 짜리 transport overhead 를 잴 수 없다(2.7초 vs 12.5ms).
> `mcp err > 0` 이면 그 run 의 재검색 결과는 **전부 무효**다 —
> tool 이 실패하면 patchcore nd 가 계산되지 않아 Δnd 가 0 으로 나온다.
"""


# ============================================================
# 그래프
# ============================================================

def plot_quant(runs: List[Dict], idx: List[Dict], out: Path) -> None:
    sel = pick(runs, "c-quant")
    if not MPL_OK or not sel:
        return
    imap = index_map(idx)
    labels = [r["run_id"].replace("c-quant-", "") for r in sel]
    auroc = [num(r.get("distance_auroc")) or 0 for r in sel]
    clsacc = [num(r.get("cls_nd_acc")) or 0 for r in sel]
    ws = [num((imap.get((r.get("index_id"), "100"), {}) or {}).get("working_set_mb")) or 0
          for r in sel]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = range(len(labels))
    w = 0.38
    ax1.bar([i - w / 2 for i in x], auroc, w, label="dist AUROC (ranking)", color="#2E7D32")
    ax1.bar([i + w / 2 for i in x], clsacc, w, label="cls@0.42 acc (fixed threshold)", color="#C62828")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels, rotation=15)
    ax1.set_ylim(0, 1); ax1.set_ylabel("score"); ax1.legend(fontsize=8)
    ax1.set_title("Ranking metric vs Fixed-threshold metric")
    ax1.grid(axis="y", alpha=.3)

    ax2.bar(labels, ws, color="#4C78A8")
    for i, v in enumerate(ws):
        ax2.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("vector working set (MB)")
    ax2.set_title("Vector working set @ coreset 100%")
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(axis="y", alpha=.3)
    fig.tight_layout()
    savefig(fig, out, "t1c_quant.png")


def plot_pareto(runs: List[Dict], idx: List[Dict], out: Path) -> None:
    if not MPL_OK:
        return
    imap = index_map(idx)
    pts = []
    for r in runs:
        x, y = num(r.get("lat_total_p95")), num(r.get("distance_auroc"))
        if x is None or y is None:
            continue
        key = (r.get("index_id"), str(r.get("vec_index_coreset_pct")))
        ws = num((imap.get(key, {}) or {}).get("working_set_mb")) or 5.0
        pts.append((x, y, ws, r["run_id"], r.get("run_id", "").split("-")[0]))
    if not pts:
        return
    colors = {"a": "#4C78A8", "b": "#F58518", "c": "#54A24B",
              "d": "#E45756", "e": "#B279A2", "f": "#72B7B2", "g": "#FF9DA6"}
    fig, ax = plt.subplots(figsize=(9, 6))
    seen = set()
    for x, y, s, rid, g in pts:
        lbl = g if g not in seen else None
        seen.add(g)
        ax.scatter(x, y, s=max(40, min(s * 6, 600)), alpha=.75,
                   c=colors.get(g, "#888"), edgecolors="k", linewidths=.5, label=lbl)
        ax.annotate(rid, (x, y), fontsize=6.5, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xlabel("end-to-end P95 latency (ms)")
    ax.set_ylabel("distance AUROC")
    ax.set_title("Pareto: latency vs detection (marker size = working set MB)")
    ax.grid(alpha=.3); ax.legend(fontsize=8, title="group")
    fig.tight_layout()
    savefig(fig, out, "t1e_pareto.png")


def plot_numcand(runs: List[Dict], out: Path) -> None:
    """recall–latency 곡선 (이중 축). knee 를 눈으로 찾는 그림."""
    sel = sorted(pick(runs, "e-numcand"),
                 key=lambda r: num(r.get("knn_num_candidates")) or 0)
    if not MPL_OK or len(sel) < 2:
        return
    x = [num(r.get("knn_num_candidates")) or 0 for r in sel]
    rec = [num(r.get("candidate_recall5")) or 0 for r in sel]
    lat = [num(r.get("lat_base_p95")) or 0 for r in sel]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, rec, "o-", color="#2E7D32", label="cand recall@5")
    ax.set_xlabel("num_candidates"); ax.set_ylabel("recall", color="#2E7D32")
    ax.set_ylim(0, 1.05); ax.grid(alpha=.3)
    ax2 = ax.twinx()
    ax2.plot(x, lat, "s--", color="#C62828", label="base P95 (ms)")
    ax2.set_ylabel("base P95 (ms)", color="#C62828")
    ax.set_title("num_candidates: recall vs latency")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="lower right")
    fig.tight_layout()
    savefig(fig, out, "t2e_numcand.png")


def plot_mcp(runs: List[Dict], out: Path) -> None:
    """MCP transport 3-axis overhead."""
    order = {"direct": 0, "mcp": 1, "mcpnative": 2}
    sel = sorted(pick(runs, "f-mcp"),
                 key=lambda r: order.get(r["run_id"].replace("f-mcp-", ""), 9))
    if not MPL_OK or not sel:
        return
    labels = [r["run_id"].replace("f-mcp-", "") for r in sel]
    mean = [num(r.get("mcp_tool_delta_mean")) or 0 for r in sel]
    p95 = [num(r.get("mcp_tool_delta_p95")) or 0 for r in sel]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xr = range(len(labels)); w = 0.38
    ax.bar([i - w / 2 for i in xr], mean, w, label="mean", color="#4C78A8")
    ax.bar([i + w / 2 for i in xr], p95, w, label="P95", color="#F58518")
    if mean and mean[0]:
        ax.axhline(mean[0], ls="--", c="gray", lw=1, label="direct baseline")
    ax.set_xticks(list(xr)); ax.set_xticklabels(labels)
    ax.set_ylabel("tool delta (ms)"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    ax.set_title("MCP transport overhead (vs direct)")
    fig.tight_layout()
    savefig(fig, out, "t3f_mcp.png")


def plot_locust(loc: List[Dict], out: Path) -> None:
    """부하 단계별 P50/P95/P99 + RPS. 포화점을 눈으로 찾는 그림."""
    if not MPL_OK or not loc:
        return
    byrun: Dict = {}
    for r in loc:
        byrun.setdefault(r.get("run_id", "?"), []).append(r)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for rid, rows in sorted(byrun.items()):
        rows = sorted(rows, key=lambda x: num(x.get("users")) or 0)
        x = [num(r.get("users")) or 0 for r in rows]
        for key, style in (("p50", "o-"), ("p95", "s--"), ("p99", "^:")):
            y = [num(r.get(key)) or 0 for r in rows]
            ax1.plot(x, y, style, label=f"{rid} {key}")
        ax2.plot(x, [num(r.get("rps")) or 0 for r in rows], "o-", label=rid)
    ax1.set_xlabel("concurrent users"); ax1.set_ylabel("latency (ms)")
    ax1.set_title("Latency vs load"); ax1.grid(alpha=.3); ax1.legend(fontsize=7)
    ax2.set_xlabel("concurrent users"); ax2.set_ylabel("RPS")
    ax2.set_title("Throughput vs load"); ax2.grid(alpha=.3); ax2.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, out, "t6_locust.png")


def plot_latency_waterfall(runs: List[Dict], out: Path) -> None:
    if not MPL_OK or not runs:
        return
    sel = [r for r in runs if num(r.get("lat_base_mean")) is not None]
    if not sel:
        return
    sel = sorted(sel, key=lambda r: r["run_id"])
    labels = [r["run_id"] for r in sel]
    base = [num(r.get("lat_base_mean")) or 0 for r in sel]
    tool = [num(r.get("lat_tool_mean")) or 0 for r in sel]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 4.5))
    ax.bar(labels, base, label="router+retriever (base)", color="#4C78A8")
    ax.bar(labels, tool, bottom=base, label="analyzer+planner+tool", color="#F58518")
    ax.set_ylabel("mean latency (ms)"); ax.legend(fontsize=8)
    ax.set_title("Latency breakdown")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    savefig(fig, out, "t1f_latency.png")


# ============================================================
# main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollup", default="experiments/results/_rollup")
    args = ap.parse_args()

    rollup = Path(args.rollup)
    runs = read_csv(rollup / "runs_wide.csv")
    idx = read_csv(rollup / "index_stats.csv")
    loc = read_csv(rollup / "locust.csv")
    prom = read_csv(rollup / "prom.csv")
    if not runs:
        print(f"runs_wide.csv 없음/비어있음: {rollup}  → collect_runs.py 먼저 실행")
        return

    tdir, pdir = rollup / "tables", rollup / "plots"
    tdir.mkdir(parents=True, exist_ok=True)

    sections = [
        ("t1a_percolate", "T1-A · Percolate 최적화 (CV C1)", table_percolate(runs)),
        ("t1b_coreset",   "T1-B · Coreset 압축 (CV C2)",     table_coreset(runs, idx)),
        ("t1c_quant",     "T1-C · Vector Quantization (CV C3)", table_quant(runs, idx)),
        ("t1d_research",  "T1-D · 재검색 인과 분해",           table_research(runs)),
        ("t1d_diag",      "T1-D-diag · 재검색 nd 갱신 진단",    table_research_diag(runs)),
        ("t2d_hnsw",      "T2-D · HNSW m/ef sweep (CV C7)",     table_hnsw(runs, idx)),
        ("t2e_numcand",   "T2-E · num_candidates 곡선",         table_numcand(runs)),
        ("t2i_bucket",    "T2-I · bucket prefilter 튜닝",        table_bucket(runs)),
        ("t3f_mcp",       "T3-F · MCP transport 3-axis (CV C4)", table_mcp(runs)),
        ("t3g_toolgate",  "T3-G · conditional re-retrieval (CV C6)", table_toolgate(runs)),
        ("t4h_percshard", "T4-H · percolator 샤드 분산",         table_percshard(runs)),
        ("t6_locust",     "T6 · 부하 3단계 (light/medium/heavy)", table_locust(loc)),
        ("p1_prom_memory","P1 · Prometheus 메모리 병목 (page fault)", table_prom_memory(prom, runs, idx)),
        ("p2_prom_llm",   "P2 · Prometheus MCP transport · TTFT/TPOT", table_prom_llm(prom, runs)),
    ]
    for name, title, body in sections:
        (tdir / f"{name}.md").write_text(f"### {title}\n\n{body}", encoding="utf-8")
        print(f"  [table] {tdir / (name + '.md')}")

    print()
    if MPL_OK:
        plot_quant(runs, idx, pdir)
        plot_pareto(runs, idx, pdir)
        plot_numcand(runs, pdir)
        plot_mcp(runs, pdir)
        plot_locust(loc, pdir)
        plot_latency_waterfall(runs, pdir)
    else:
        print(_mpl_hint())

    # 전체 묶음
    summ = {}
    sp = rollup / "collect_summary.json"
    if sp.exists():
        try:
            summ = json.load(io.open(sp, encoding="utf-8"))
        except Exception:
            summ = {}
    parts = ["# 실험 결과 — Tier 1\n",
             f"- runs: **{len(runs)}**  ·  indices: **{len(idx)}**\n"]
    if summ.get("skipped"):
        parts.append(f"- 제외: {', '.join(summ['skipped'])}\n")
    parts.append("")
    for name, title, body in sections:
        parts.append(f"### {title}\n\n{body}")
        png = pdir / f"{name}.png"
        if png.exists():
            parts.append(f"\n![{name}](plots/{name}.png)\n")
    for extra in ("t2e_numcand", "t3f_mcp", "t6_locust", "t1e_pareto", "t1f_latency"):
        if (pdir / f"{extra}.png").exists():
            parts.append(f"\n### {extra}\n\n![{extra}](plots/{extra}.png)\n")
    (rollup / "results.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"\n  [report] {rollup / 'results.md'}")


if __name__ == "__main__":
    main()

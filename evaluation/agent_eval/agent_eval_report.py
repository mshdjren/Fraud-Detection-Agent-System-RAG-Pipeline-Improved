"""
agent_eval_report.py — 수치 / 표 / 그래프 생성
=================================================
agent_batch_inference.py가 저장한 agent_eval_report.json을 읽어
포트폴리오용 수치 요약, 비교 표, 그래프를 출력한다.

생성 산출물:
  (A) 수치 요약   — 콘솔 출력 + summary_table.csv
  (B) 비교 표     — method_comparison.csv
  (C) 그래프 4종  — PNG 파일
      1. roc_curve_comparison.png  — distance vs agent ROC 곡선
      2. boundary_fnr_bar.png      — boundary/non-boundary FNR 비교
      3. latency_breakdown.png     — 구간별 stacked bar + box
      4. action_distribution.png  — expand/reroute/accept 파이 차트

Usage:
  python agent_eval_report.py \\
    --report-path /results/agent_eval/agent_eval_report.json \\
    --output-dir  /results/agent_eval/plots
"""

from __future__ import annotations

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")   # GKE/headless 환경
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MPL_OK = True
except ImportError:
    MPL_OK = False

try:
    import pandas as pd
    PD_OK = True
except ImportError:
    PD_OK = False


# ─── 공통 스타일 ────────────────────────────────────────────
COLORS = {
    "distance":    "#4C78A8",
    "agent_rag":   "#F58518",
    "boundary":    "#E45756",
    "non_boundary":"#72B7B2",
    "expand":      "#54A24B",
    "reroute":     "#EECA3B",
    "accept":      "#B279A2",
    "base":        "#9ECAE9",
    "tool_delta":  "#AEC7E8",
}


# ============================================================
# A. 수치 요약 출력
# ============================================================

def print_summary(report: Dict) -> None:
    """핵심 수치를 콘솔에 일목요연하게 출력한다."""
    s = report.get("agent_summary", {})
    thr = report.get("threshold_info", {})
    bnd = report.get("boundary", {})
    vcr = report.get("verdict_change", {})
    act = report.get("action_distribution", {})
    lat = report.get("latency_breakdown", {})

    print("\n" + "=" * 65)
    print(f"  Agent Loop Evaluation — {report.get('experiment_case', '')}")
    print("=" * 65)

    print("\n[1] Detection Performance")
    print(f"  distance AUROC          : {report.get('dist_auroc', {}).get('distance_auroc')}")
    print(f"  agent AUROC (binary)    : {report.get('anomaly_auroc', {}).get('auroc')}")
    print(f"  best_threshold (Youden) : {thr.get('best_threshold')}  "
          f"(FPR={thr.get('best_fpr')}, FNR={thr.get('best_fnr')})")

    print("\n[2] @ Fixed Threshold Comparison")
    d_row = report.get("distance_at_threshold", {})
    a_row = report.get("agent_at_threshold", {})
    print(f"  {'Method':<20} {'FNR':>6} {'FPR':>6} {'F1':>6} {'Acc':>6}")
    print(f"  {'-'*48}")
    print(f"  {'Distance-only':<20} {d_row.get('fnr','-'):>6} {d_row.get('fpr','-'):>6} "
          f"{d_row.get('f1','-'):>6} {d_row.get('accuracy','-'):>6}")
    print(f"  {'Agent RAG':<20} {a_row.get('fnr','-'):>6} {a_row.get('fpr','-'):>6} "
          f"{a_row.get('f1','-'):>6} {a_row.get('accuracy','-'):>6}")

    print("\n[3] Boundary Zone (top1_dist ∈ [0.30, 0.55))")
    print(f"  boundary samples        : {bnd.get('boundary_n')} "
          f"(fraud={bnd.get('boundary_fraud_n')})")
    print(f"  boundary FNR            : {bnd.get('boundary_fnr')}  "
          f"← 핵심 지표 (낮을수록 agent loop 기여)")
    print(f"  non-boundary FNR        : {bnd.get('non_boundary_fnr')}")
    print(f"  expand rate (boundary)  : {bnd.get('boundary_expand_rate')}")
    print(f"  verdict change rate     : {bnd.get('boundary_verdict_change_rate')}")

    print("\n[4] Agent Action Distribution")
    print(f"  fast track              : {act.get('fast_rate')} ({act.get('fast_track_n')}건)")
    print(f"  expand_within_cluster   : {act.get('expand_rate')} ({act.get('expand_n')}건)")
    print(f"  reroute_cross_cluster   : {act.get('reroute_rate')} ({act.get('reroute_n')}건)")
    print(f"  accept (1st round)      : {act.get('deep_track_n', 0) - act.get('expand_n', 0) - act.get('reroute_n', 0)}건")

    print("\n[5] Verdict Change")
    print(f"  total with tool         : {vcr.get('n_with_tool')}")
    print(f"  verdict changed         : {vcr.get('n_changed')} ({vcr.get('verdict_change_rate')})")
    print(f"  correct flip (FN→TP)   : {vcr.get('correct_flip_n')} ({vcr.get('correct_flip_rate')})")
    print(f"  wrong flip  (TP→FN)   : {vcr.get('wrong_flip_n')}")

    lat_bd = lat or report.get("latency_breakdown", {})
    total_s = lat_bd.get("total_ms", {})
    base_s  = lat_bd.get("base_retrieve_ms", {})
    tool_s  = lat_bd.get("tool_delta_ms", {})
    print("\n[6] Latency")
    print(f"  total     P50={total_s.get('p50')}ms  P95={total_s.get('p95')}ms")
    print(f"  base_ret  P50={base_s.get('p50')}ms   P95={base_s.get('p95')}ms")
    print(f"  tool_Δ    P50={tool_s.get('p50')}ms   P95={tool_s.get('p95')}ms")
    print("=" * 65)


# ============================================================
# B. 비교 표 CSV 저장
# ============================================================

def save_comparison_table(report: Dict, out_dir: Path) -> Optional[Path]:
    if not PD_OK:
        print("[skip] pandas 없음 — CSV 저장 건너뜀")
        return None

    d_row = report.get("distance_at_threshold", {})
    a_row = report.get("agent_at_threshold", {})
    thr   = report.get("threshold_info", {})

    rows = [
        {
            "method":    "Distance-only RAG",
            "auroc":     report.get("dist_auroc", {}).get("distance_auroc"),
            "threshold": thr.get("best_threshold"),
            "fnr":       d_row.get("fnr"),
            "fpr":       d_row.get("fpr"),
            "precision": d_row.get("precision"),
            "recall":    d_row.get("recall"),
            "f1":        d_row.get("f1"),
            "accuracy":  d_row.get("accuracy"),
        },
        {
            "method":    "Agent RAG (LLM + loop)",
            "auroc":     report.get("anomaly_auroc", {}).get("auroc"),
            "threshold": "binary@50",
            "fnr":       a_row.get("fnr"),
            "fpr":       a_row.get("fpr"),
            "precision": a_row.get("precision"),
            "recall":    a_row.get("recall"),
            "f1":        a_row.get("f1"),
            "accuracy":  a_row.get("accuracy"),
        },
    ]
    df   = pd.DataFrame(rows)
    path = out_dir / "method_comparison.csv"
    df.to_csv(path, index=False)
    print(f"[saved] {path}")
    return path


def save_summary_csv(report: Dict, out_dir: Path) -> Optional[Path]:
    if not PD_OK:
        return None
    s   = report.get("agent_summary", {})
    bnd = report.get("boundary", {})
    vcr = report.get("verdict_change", {})
    act = report.get("action_distribution", {})

    rows = [
        {"metric": "distance_auroc",         "value": report.get("dist_auroc", {}).get("distance_auroc")},
        {"metric": "agent_auroc",             "value": report.get("anomaly_auroc", {}).get("auroc")},
        {"metric": "best_threshold",          "value": report.get("threshold_info", {}).get("best_threshold")},
        {"metric": "distance_fnr@thr",        "value": report.get("distance_at_threshold", {}).get("fnr")},
        {"metric": "agent_fnr@thr",           "value": report.get("agent_at_threshold", {}).get("fnr")},
        {"metric": "boundary_fnr",            "value": bnd.get("boundary_fnr")},
        {"metric": "non_boundary_fnr",        "value": bnd.get("non_boundary_fnr")},
        {"metric": "boundary_expand_rate",    "value": bnd.get("boundary_expand_rate")},
        {"metric": "verdict_change_rate",     "value": vcr.get("verdict_change_rate")},
        {"metric": "correct_flip_rate",       "value": vcr.get("correct_flip_rate")},
        {"metric": "expand_rate",             "value": act.get("expand_rate")},
        {"metric": "reroute_rate",            "value": act.get("reroute_rate")},
        {"metric": "latency_p95_total_ms",    "value": s.get("latency_total_p95_ms")},
    ]
    df   = pd.DataFrame(rows)
    path = out_dir / "summary_table.csv"
    df.to_csv(path, index=False)
    print(f"[saved] {path}")
    return path


# ============================================================
# C-1. ROC Curve Comparison
# ============================================================

def plot_roc_curve(report: Dict, out_dir: Path) -> None:
    if not MPL_OK:
        return
    thr = report.get("threshold_info", {})
    if "_fpr_arr" not in thr:
        print("[skip] ROC 곡선 데이터 없음")
        return

    fpr = np.array(thr["_fpr_arr"])
    tpr = np.array(thr["_tpr_arr"])
    dist_auroc  = report.get("dist_auroc", {}).get("distance_auroc")
    agent_auroc = report.get("anomaly_auroc", {}).get("auroc")
    best_fpr    = thr.get("best_fpr")
    best_tpr    = thr.get("best_tpr")

    fig, ax = plt.subplots(figsize=(7, 6))

    # Distance ROC 곡선
    label_d = f"Distance RAG  (AUROC={dist_auroc})"
    ax.plot(fpr, tpr, color=COLORS["distance"], lw=2, label=label_d)

    # Agent RAG 단일 점 (binary, threshold=50 고정)
    a_row = report.get("agent_at_threshold", {})
    a_fpr = a_row.get("fpr")
    a_tpr = a_row.get("recall")  # TPR = recall
    if a_fpr is not None and a_tpr is not None:
        label_a = f"Agent RAG  (AUROC≈{agent_auroc})"
        ax.scatter([a_fpr], [a_tpr], color=COLORS["agent_rag"], s=120,
                   zorder=5, label=label_a, marker="*")

    # Best threshold 표시
    if best_fpr is not None and best_tpr is not None:
        ax.scatter([best_fpr], [best_tpr], color="red", s=80,
                   zorder=6, label=f"Best thr={thr.get('best_threshold')}", marker="D")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax.set_title("ROC Curve — Distance RAG vs Agent RAG", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.grid(alpha=0.3)

    path = out_dir / "roc_curve_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


# ============================================================
# C-2. Boundary FNR Bar Chart
# ============================================================

def plot_boundary_fnr(report: Dict, out_dir: Path) -> None:
    if not MPL_OK:
        return
    bnd = report.get("boundary", {})
    b_fnr  = bnd.get("boundary_fnr")
    nb_fnr = bnd.get("non_boundary_fnr")
    d_row  = report.get("distance_at_threshold", {})
    a_row  = report.get("agent_at_threshold", {})

    labels = ["Non-boundary\n(distance only)", "Boundary\n(distance only)",
              "Non-boundary\n(agent RAG)",     "Boundary\n(agent RAG)"]
    values = [nb_fnr, b_fnr,
              d_row.get("fnr"),                # 전체 FNR 참고용
              a_row.get("fnr")]
    colors = [COLORS["distance"], COLORS["boundary"],
              COLORS["agent_rag"], COLORS["boundary"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, values):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, float(val) + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("False Negative Rate (FNR)\n← 낮을수록 좋음", fontsize=11)
    ax.set_title("Boundary Zone FNR — Distance RAG vs Agent RAG\n"
                 "(boundary: top1_dist ∈ [0.30, 0.55))", fontsize=12)
    ax.set_ylim(0, max(v for v in values if v is not None) * 1.3)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.3, color="gray", linestyle="--", alpha=0.5, label="FNR=0.3 기준선")
    ax.legend(fontsize=9)

    path = out_dir / "boundary_fnr_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


# ============================================================
# C-3. Latency Breakdown
# ============================================================

def plot_latency_breakdown(report: Dict, out_dir: Path) -> None:
    if not MPL_OK:
        return
    lat = report.get("latency_breakdown", {})
    base  = lat.get("base_retrieve_ms", {})
    tool  = lat.get("tool_delta_ms", {})
    total = lat.get("total_ms", {})

    # ── 왼쪽: stacked bar (P50/P95) ──────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    categories = ["P50", "P95"]
    base_vals  = [base.get("p50", 0),  base.get("p95", 0)]
    tool_vals  = [tool.get("p50", 0),  tool.get("p95", 0)]
    x          = np.arange(len(categories))

    b1 = ax1.bar(x, base_vals, label="base_retrieve",  color=COLORS["base"],       alpha=0.85)
    b2 = ax1.bar(x, tool_vals, bottom=base_vals,        label="tool_Δ (analyzer+planner+tool)",
                 color=COLORS["tool_delta"], alpha=0.85)
    for bar, bv, tv in zip(b1, base_vals, tool_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 (bv + tv) + 5, f"{bv + tv:.0f}ms",
                 ha="center", va="bottom", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=12)
    ax1.set_ylabel("Latency (ms)", fontsize=11)
    ax1.set_title("Agent Loop Latency Breakdown\n(stacked)", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # ── 오른쪽: boxplot (raw 데이터) ─────────────────────────
    base_arr  = np.array(base.get("_arr",  []))
    tool_arr  = np.array(tool.get("_arr",  []))
    total_arr = np.array(total.get("_arr", []))

    if len(total_arr) > 0:
        bp_data   = []
        bp_labels = []
        if len(base_arr) > 0:
            bp_data.append(base_arr); bp_labels.append("base_retrieve")
        if len(tool_arr) > 0:
            bp_data.append(tool_arr); bp_labels.append("tool_Δ")
        if len(total_arr) > 0:
            bp_data.append(total_arr); bp_labels.append("total")

        bp = ax2.boxplot(bp_data, patch_artist=True,
                         medianprops={"color": "black", "linewidth": 2})
        colors_bp = [COLORS["base"], COLORS["tool_delta"], COLORS["distance"]]
        for patch, c in zip(bp["boxes"], colors_bp[:len(bp_data)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax2.set_xticklabels(bp_labels, fontsize=10)
        ax2.set_ylabel("Latency (ms)", fontsize=11)
        ax2.set_title("Latency Distribution\n(boxplot)", fontsize=12)
        ax2.grid(axis="y", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "raw 데이터 없음", transform=ax2.transAxes,
                 ha="center", va="center", fontsize=12)

    fig.tight_layout()
    path = out_dir / "latency_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


# ============================================================
# C-4. Action Distribution
# ============================================================

def plot_action_distribution(report: Dict, out_dir: Path) -> None:
    if not MPL_OK:
        return
    act = report.get("action_distribution", {})

    expand_n  = act.get("expand_n", 0)
    reroute_n = act.get("reroute_n", 0)
    fast_n    = act.get("fast_track_n", 0)
    deep_n    = act.get("deep_track_n", 0)
    accept_n  = max(deep_n - expand_n - reroute_n, 0)

    labels = ["Fast Track",
              "Deep: accept\n(1st round confident)",
              "Deep: expand\n_within_cluster",
              "Deep: reroute\n_cross_cluster"]
    sizes  = [fast_n, accept_n, expand_n, reroute_n]
    colors = ["#AEC7E8", COLORS["accept"], COLORS["expand"], COLORS["reroute"]]

    # 0인 항목 제거
    non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
    if not non_zero:
        print("[skip] action distribution 데이터 없음")
        return
    labels_, sizes_, colors_ = zip(*non_zero)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 파이 차트
    wedges, texts, autotexts = ax1.pie(
        sizes_, labels=labels_, colors=colors_,
        autopct="%1.1f%%", pctdistance=0.85,
        startangle=140, counterclock=False,
    )
    for text in autotexts:
        text.set_fontsize(10)
    ax1.set_title("Agent Action Distribution (전체)", fontsize=12)

    # 막대 차트 (건수)
    x = np.arange(len(labels_))
    ax2.bar(x, sizes_, color=colors_, alpha=0.85, edgecolor="white")
    for xi, si in zip(x, sizes_):
        ax2.text(xi, si + 0.3, str(si), ha="center", va="bottom", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_, fontsize=9)
    ax2.set_ylabel("건수", fontsize=11)
    ax2.set_title("Agent Action Distribution (건수)", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = out_dir / "action_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


# ============================================================
# Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Agent Evaluation Report — 수치/표/그래프 생성",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--report-path", type=str,
                        default="./results/agent_eval/agent_eval_report.json")
    parser.add_argument("--output-dir",  type=str,
                        default="./results/agent_eval/plots")
    parser.add_argument("--no-plots",    action="store_true",
                        help="그래프 생성 건너뜀 (수치/표만 출력)")
    args = parser.parse_args()

    report  = json.loads(Path(args.report_path).read_text())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── A. 수치 요약 ─────────────────────────────────────────
    print_summary(report)

    # ── B. CSV 표 ────────────────────────────────────────────
    save_comparison_table(report, out_dir)
    save_summary_csv(report, out_dir)

    # ── C. 그래프 ────────────────────────────────────────────
    if not args.no_plots:
        if MPL_OK:
            plot_roc_curve(report, out_dir)
            plot_boundary_fnr(report, out_dir)
            plot_latency_breakdown(report, out_dir)
            plot_action_distribution(report, out_dir)
        else:
            print("[skip] matplotlib 없음 — 그래프 생성 건너뜀")

    print(f"\n[완료] 모든 산출물 → {out_dir}")


if __name__ == "__main__":
    main()

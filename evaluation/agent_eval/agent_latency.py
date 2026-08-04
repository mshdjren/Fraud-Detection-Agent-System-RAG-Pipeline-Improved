"""
agent_latency.py — Agent Loop Latency 분석 전용
================================================
batch inference 결과(agent_predictions.json)에서 latency를 분해하고
MCP 3-axis transport overhead를 측정한다.

측정 대상:
  1. End-to-end P50/P95 — track별(fast/deep/deep+tool) 분리
  2. Segment breakdown — base_retrieve / tool_delta 구간
  3. MCP overhead — direct vs mcp vs mcp_native 모드별 tool_delta 비교
     (같은 샘플셋을 세 모드로 각각 실행해야 정확한 비교 가능)

Locust vs batch_inference 측정 기준:
  - MCP overhead(~12.5ms) 측정: batch_inference 방식 권장
    (순차 단건 호출 → 각 요청의 정확한 wall-clock breakdown 가능)
  - Saturation point(동시 N명): Locust 권장
    (현 포트폴리오 범위에서는 개념 언급 수준으로 충분)
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================
# 1. Latency 데이터 파싱
# ============================================================

def parse_latency_records(predictions: List[Dict]) -> Dict[str, List[float]]:
    """
    predictions에서 latency 구간별 배열을 추출한다.

    반환 키:
      total_ms: 전체 end-to-end
      base_retrieve_ms: orchestrator의 base retriever 호출
      tool_delta_ms: total - base_retrieve (analyzer+planner+tool)
      client_ms: 클라이언트 측정 latency
    """
    records: Dict[str, List[float]] = {
        "total_ms": [], "base_retrieve_ms": [],
        "tool_delta_ms": [], "client_ms": [],
        "fast_total_ms": [], "deep_total_ms": [],
        "deep_with_tool_ms": [], "deep_no_tool_ms": [],
        "ttft_ms": [], "tpot_ms": [], "out_tokens": [],
    }

    for pred in predictions:
        if pred.get("error"):
            continue
        lms   = pred.get("latency_ms", {})
        track = pred.get("track", "deep")
        trace = pred.get("agent_trace", [])
        has_tool = any(
            "expand_within_cluster" in s or "reroute_cross_cluster" in s
            for s in trace
        )

        if not isinstance(lms, dict):
            continue

        _ah = pred.get("analyzer_history") or []
        if _ah and isinstance(_ah[-1], dict):
            for _src, _dst in (("ttft_ms", "ttft_ms"), ("tpot_ms", "tpot_ms"),
                               ("n_output_tokens", "out_tokens")):
                _v = _ah[-1].get(_src)
                if _v is not None:
                    records[_dst].append(float(_v))
    
        total = lms.get("total")
        base  = lms.get("base_retrieve")
        cli   = pred.get("_client_latency_ms")

        if total is not None:
            records["total_ms"].append(float(total))
            if track == "fast":
                records["fast_total_ms"].append(float(total))
            else:
                records["deep_total_ms"].append(float(total))
                if has_tool:
                    records["deep_with_tool_ms"].append(float(total))
                else:
                    records["deep_no_tool_ms"].append(float(total))

        if base is not None:
            records["base_retrieve_ms"].append(float(base))
            if total is not None:
                records["tool_delta_ms"].append(float(total) - float(base))

        if cli is not None:
            records["client_ms"].append(float(cli))

    return records


# ============================================================
# 2. 통계 요약
# ============================================================

def summarize_latency(records: Dict[str, List[float]]) -> Dict:
    """각 구간의 mean/p50/p95/p99를 계산한다."""
    def _stats(arr: List[float]) -> Dict:
        if not arr:
            return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None}
        a = np.array(arr)
        return {
            "n":    len(a),
            "mean": round(float(np.mean(a)), 2),
            "p50":  round(float(np.percentile(a, 50)), 2),
            "p95":  round(float(np.percentile(a, 95)), 2),
            "p99":  round(float(np.percentile(a, 99)), 2),
            "_arr": arr,  # 그래프용 raw 데이터
        }

    return {segment: _stats(vals) for segment, vals in records.items()}


# ============================================================
# 3. MCP 3-axis Overhead 비교 테이블
# ============================================================

def compare_mcp_modes(
    direct_preds:     List[Dict],
    mcp_preds:        List[Dict],
    mcp_native_preds: Optional[List[Dict]] = None,
) -> Dict:
    """
    동일 샘플셋을 세 모드(direct/mcp/mcp_native)로 실행한 결과를 비교.
    tool_delta_ms(= total - base_retrieve)가 transport overhead를 포함한 구간.

    이상적으로는 같은 N개 샘플에 대해 모드만 바꿔 세 번 실행해야 한다.

    반환:
      comparison_table: 세 모드의 tool_delta 통계 비교
      mcp_overhead_ms: mcp - direct (estimated overhead)
      mcp_native_overhead_ms: mcp_native - direct
    """
    def _tool_deltas(preds):
        deltas = []
        for p in preds:
            if p.get("error"):
                continue
            lms = p.get("latency_ms", {})
            if isinstance(lms, dict):
                total = lms.get("total")
                base  = lms.get("base_retrieve")
                if total is not None and base is not None:
                    deltas.append(float(total) - float(base))
        return deltas

    direct_delta     = _tool_deltas(direct_preds)
    mcp_delta        = _tool_deltas(mcp_preds)
    native_delta     = _tool_deltas(mcp_native_preds or [])

    def _mean(arr): return round(float(np.mean(arr)), 2) if arr else None
    def _p95(arr):  return round(float(np.percentile(arr, 95)), 2) if arr else None

    table = {
        "direct":     {"mean_ms": _mean(direct_delta),  "p95_ms": _p95(direct_delta),  "n": len(direct_delta)},
        "mcp":        {"mean_ms": _mean(mcp_delta),     "p95_ms": _p95(mcp_delta),     "n": len(mcp_delta)},
        "mcp_native": {"mean_ms": _mean(native_delta),  "p95_ms": _p95(native_delta),  "n": len(native_delta)},
    }
    # overhead 추정 (mean 기준)
    mcp_overhead        = None
    mcp_native_overhead = None
    if _mean(direct_delta) and _mean(mcp_delta):
        mcp_overhead = round(_mean(mcp_delta) - _mean(direct_delta), 2)
    if _mean(direct_delta) and _mean(native_delta):
        mcp_native_overhead = round(_mean(native_delta) - _mean(direct_delta), 2)

    return {
        "comparison_table":      table,
        "mcp_overhead_ms":       mcp_overhead,
        "mcp_native_overhead_ms": mcp_native_overhead,
        # 그래프용 raw 데이터
        "_direct_deltas":     direct_delta,
        "_mcp_deltas":        mcp_delta,
        "_mcp_native_deltas": native_delta,
    }


# ============================================================
# 4. Latency 보고서 저장
# ============================================================

def save_latency_report(
    predictions: List[Dict],
    out_dir: Path,
    mode: str = "direct",
) -> Dict:
    """
    단일 모드 실행 결과의 latency를 분석하고 저장한다.
    MCP 3-axis 비교는 compare_mcp_modes()를 별도 호출해야 한다.
    """
    records = parse_latency_records(predictions)
    summary = summarize_latency(records)

    report = {
        "mode":    mode,
        "summary": summary,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"latency_{mode}.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    return report

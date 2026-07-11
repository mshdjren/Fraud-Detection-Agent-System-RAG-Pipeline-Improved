"""
agent_eval_metrics.py — Agent Loop 전용 Evaluation Metrics
===========================================================
evaluation_metrics.py(forward RAG)를 재사용하면서 agent loop 특화
지표를 추가한다.

새로 정의하는 지표:
  1. LLM Binary 판단 → distance best-threshold와 동일 기준으로 정렬
  2. Boundary Zone FNR — top1_distance 0.3~0.55 구간만 분리 측정
  3. Verdict Change Rate — expand/reroute 후 판정이 바뀐 비율
  4. Action Distribution — expand/reroute/accept 비율
  5. Agent Latency Breakdown — base_retrieve / tool / total 구간

의존성:
  evaluation_metrics.py (기존 MetricCalculator)
  sklearn (AUROC, roc_curve)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
import structlog

try:
    from sklearn.metrics import (
        roc_auc_score, roc_curve,
        confusion_matrix, classification_report,
    )
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# 기존 MetricCalculator 재사용
from evaluation_metrics import MetricCalculator

logger = structlog.get_logger()

# ── Boundary zone 정의 — serving(local_analyzer/controller)과 동일 env를
# 읽어 단일 소스로 통일 (기존 하드코딩 0.30/0.55는 env 오버라이드 시
# serving과 eval의 band가 어긋나는 위험이 있었음) ──
import os
BOUNDARY_LOW  = float(os.getenv("SUFFICIENCY_BOUNDARY_LOW",  "0.3"))
BOUNDARY_HIGH = float(os.getenv("SUFFICIENCY_BOUNDARY_HIGH", "0.55"))


# ============================================================
# 1. LLM Score 변환 — distance AUROC와 공통 축으로 정렬
# ============================================================

def build_llm_scores(predictions: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    LLM 이진 판단(NORMAL/ABNORMAL)을 anomaly score(0~100)로 변환.
    AUROC 계산 시 distance score와 동일 방향(높을수록 이상)으로 맞춘다.

    변환 규칙:
      ABNORMAL → confidence 그대로 (높을수록 이상)
      NORMAL   → 100 - confidence (낮은 신뢰 정상 = 높은 이상 점수)

    현재 confidence=85 고정 문제:
      retuning 전에는 NORMAL/ABNORMAL 이진값(0/1)만 의미 있음.
      confidence가 가변화되면 soft score로 활용 가능.

    반환: (scores, valid_mask)
      scores: 이상 점수 배열 (0~100)
      valid_mask: 에러/결측 제외 mask
    """
    scores = []
    valid  = []
    for pred in predictions:
        if pred.get("error") or pred.get("classification") is None:
            scores.append(np.nan)
            valid.append(False)
            continue
        cls  = pred.get("classification", "NORMAL")
        conf = float(pred.get("confidence") or 85)
        score = conf if cls == "ABNORMAL" else (100.0 - conf)
        scores.append(score)
        valid.append(True)
    return np.array(scores, dtype=float), np.array(valid, dtype=bool)


# ============================================================
# 2. Best Threshold 산출 (Youden's J)
# ============================================================

def find_best_threshold(
    gt_labels: np.ndarray,
    scores: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict:
    """
    distance score의 ROC 곡선에서 Youden's J(= Sensitivity + Specificity - 1)
    최대값 기준 best threshold를 결정한다.

    이 threshold를 LLM / Agent RAG 비교 기준으로 고정하면
    세 방법이 동일한 operating point에서 비교된다.

    반환:
      best_threshold: float
      best_fpr: float
      best_tpr: float (= Recall = 1 - FNR)
      auroc: float
    """
    if not SKLEARN_OK:
        return {"error": "sklearn not available"}

    y    = gt_labels[valid_mask].astype(int)
    s    = scores[valid_mask]
    if y.sum() == 0 or y.sum() == len(y):
        return {"error": "single class — AUROC undefined"}

    auroc = float(roc_auc_score(y, s))
    fpr_arr, tpr_arr, thresh_arr = roc_curve(y, s)

    # Youden's J
    j_scores = tpr_arr - fpr_arr
    best_idx  = int(np.argmax(j_scores))

    return {
        "auroc":           round(auroc, 4),
        "best_threshold":  round(float(thresh_arr[best_idx]), 4),
        "best_fpr":        round(float(fpr_arr[best_idx]), 4),
        "best_tpr":        round(float(tpr_arr[best_idx]), 4),
        "best_fnr":        round(1.0 - float(tpr_arr[best_idx]), 4),
        "youden_j":        round(float(j_scores[best_idx]), 4),
        "n_evaluated":     int(valid_mask.sum()),
        # ROC 곡선 데이터 (그래프용)
        "_fpr_arr":        fpr_arr.tolist(),
        "_tpr_arr":        tpr_arr.tolist(),
        "_thresh_arr":     thresh_arr.tolist(),
    }


# ============================================================
# 3. Agent-Specific MetricCalculator 확장
# ============================================================

class AgentMetricCalculator(MetricCalculator):
    """
    MetricCalculator 상속 + agent loop 특화 지표 추가.
    기존 forward RAG 지표(AUROC, recall@5, latency)는 그대로 사용.
    """

    def __init__(self):
        super().__init__()
        self._agent_summary: Dict = {}

    # ── 3-1. Binary Classification @ Fixed Threshold ────────

    def calculate_classification_at_threshold(
        self,
        gt_labels:    np.ndarray,
        scores:       np.ndarray,
        valid_mask:   np.ndarray,
        threshold:    float,
        method_name:  str = "method",
    ) -> Dict:
        """
        고정 threshold에서의 FPR, FNR, Accuracy, F1을 계산.
        distance / LLM / Agent RAG 세 방법을 동일 threshold에서 비교하는 함수.

        threshold: distance 기준 best_threshold (find_best_threshold 결과값)
        scores:    anomaly score (높을수록 이상)
        """
        y    = gt_labels[valid_mask].astype(int)
        s    = scores[valid_mask]
        pred = (s >= threshold).astype(int)

        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())

        tpr = tp / max(tp + fn, 1)   # Sensitivity / Recall
        fpr = fp / max(fp + tn, 1)   # Fall-out
        fnr = fn / max(fn + tp, 1)   # Miss Rate
        acc = (tp + tn) / max(len(y), 1)
        pre = tp / max(tp + fp, 1)
        f1  = 2 * pre * tpr / max(pre + tpr, 1e-9)

        result = {
            "method":    method_name,
            "threshold": round(threshold, 4),
            "accuracy":  round(acc, 4),
            "precision": round(pre, 4),
            "recall":    round(tpr, 4),   # = TPR = 1 - FNR
            "fnr":       round(fnr, 4),
            "fpr":       round(fpr, 4),
            "f1":        round(f1, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_evaluated": int(len(y)),
        }
        self._agent_summary[f"{method_name}_fnr"]      = round(fnr, 4)
        self._agent_summary[f"{method_name}_fpr"]      = round(fpr, 4)
        self._agent_summary[f"{method_name}_accuracy"] = round(acc, 4)
        return result

    # ── 3-2. Boundary Zone 분리 측정 ────────────────────────

    def calculate_boundary_metrics(
        self,
        predictions: List[Dict],
        gt_labels:   np.ndarray,
        threshold:   float,
        boundary_low:  float = BOUNDARY_LOW,
        boundary_high: float = BOUNDARY_HIGH,
    ) -> Dict:
        """
        top1_distance가 boundary zone(low ~ high)에 있는 샘플만 분리해서
        FNR을 측정한다.

        핵심 질문: "agent loop가 경계 케이스에서 실제로 fraud를 덜 놓치는가?"

        반환:
          boundary_fnr:      boundary zone의 FNR (낮을수록 좋음)
          boundary_n:        boundary 샘플 수
          non_boundary_fnr:  non-boundary 샘플의 FNR (비교용)
          expand_rate:       boundary 샘플 중 expand 발동 비율
          verdict_change_rate: expand 후 판정 변화 비율
        """
        b_correct = b_fraud = b_expand = b_changed = 0
        nb_correct = nb_fraud = 0

        for i, pred in enumerate(predictions):
            if pred.get("error") or i >= len(gt_labels):
                continue
            dist  = float(pred.get("top_1_distance") or pred.get("top1_distance") or 0)
            label = int(gt_labels[i])
            cls   = pred.get("classification", "NORMAL")
            trace = pred.get("agent_trace", [])
            r0_cls = pred.get("_round0_classification")

            # ★ [버그 수정] 기존: band [0.3,0.55)를 raw L2(≈0.8~1.5)에 대고
            # 비교해 boundary_n=0으로 지표 전체가 무효였다. serving과 동일한
            # 판정 기준인 round0 sufficiency(=nd-band 주입값)를 1차 소스로
            # 사용하고, rounds가 없는 구버전 predictions만 raw 비교로 fallback.
            rounds = pred.get("agent_rounds") or []
            r0_suf = rounds[0].get("sufficiency") if rounds else None
            if r0_suf in ("ambiguous", "confident"):
                in_boundary = (r0_suf == "ambiguous")
            else:
                in_boundary = boundary_low <= dist < boundary_high

            if in_boundary:
                if label == 1:  # 실제 fraud
                    b_fraud += 1
                    if cls == "ABNORMAL":
                        b_correct += 1
                # expand/reroute 발동 여부
                has_tool = any(
                    s in trace for s in [
                        "expand_within_cluster:direct",
                        "reroute_cross_cluster:direct",
                        "expand_within_cluster:mcp",
                        "reroute_cross_cluster:mcp",
                        "expand_within_cluster:mcp_native",
                        "reroute_cross_cluster:mcp_native",
                    ]
                )
                if has_tool:
                    b_expand += 1
                    if r0_cls and r0_cls != cls:
                        b_changed += 1
            else:
                if label == 1 and cls == "ABNORMAL":
                    nb_correct += 1
                if label == 1:
                    nb_fraud += 1

        b_total = sum(1 for i, p in enumerate(predictions)
                      if not p.get("error") and i < len(gt_labels)
                      and boundary_low <= float(p.get("top_1_distance") or p.get("top1_distance") or 0) < boundary_high)
        nb_total = sum(1 for i, p in enumerate(predictions)
                       if not p.get("error") and i < len(gt_labels)
                       and not (boundary_low <= float(p.get("top_1_distance") or p.get("top1_distance") or 0) < boundary_high))

        b_fnr  = 1.0 - (b_correct / max(b_fraud, 1))
        nb_fnr = 1.0 - (nb_correct / max(nb_fraud, 1))

        result = {
            "boundary_range":         f"[{boundary_low}, {boundary_high})",
            "boundary_n":             b_total,
            "boundary_fraud_n":       b_fraud,
            "boundary_fnr":           round(b_fnr, 4),
            "non_boundary_n":         nb_total,
            "non_boundary_fnr":       round(nb_fnr, 4),
            "boundary_expand_rate":   round(b_expand / max(b_total, 1), 4),
            "boundary_verdict_change_rate": round(b_changed / max(b_expand, 1), 4),
            "boundary_correct_flip_n": b_changed,
        }
        self._agent_summary["boundary_fnr"]  = round(b_fnr, 4)
        self._agent_summary["boundary_expand_rate"] = round(b_expand / max(b_total, 1), 4)
        return result

    # ── 3-3. Verdict Change Rate (전체) ─────────────────────

    def calculate_verdict_change_rate(
        self, predictions: List[Dict], gt_labels: np.ndarray
    ) -> Dict:
        """
        expand/reroute 발동 후 최종 판정이 round 0 판정과 다른 비율.
        _round0_classification 필드는 batch_inference 단계에서 저장됨.

        올바른 flip (correct_flip):
          round 0: NORMAL, round 1: ABNORMAL, gt: fraud(1) → 좋은 변화
        잘못된 flip (wrong_flip):
          round 0: ABNORMAL, round 1: NORMAL, gt: fraud(1) → 나쁜 변화
        """
        total_with_tool = 0
        changed = correct_flip = wrong_flip = 0

        for i, pred in enumerate(predictions):
            if pred.get("error") or i >= len(gt_labels):
                continue
            trace = pred.get("agent_trace", [])
            has_tool = any("expand_within_cluster" in s or "reroute_cross_cluster" in s
                           for s in trace)
            if not has_tool:
                continue
            total_with_tool += 1

            r0  = pred.get("_round0_classification")
            r1  = pred.get("classification")
            lbl = int(gt_labels[i])
            if r0 and r1 and r0 != r1:
                changed += 1
                if lbl == 1 and r0 == "NORMAL" and r1 == "ABNORMAL":
                    correct_flip += 1
                elif lbl == 1 and r0 == "ABNORMAL" and r1 == "NORMAL":
                    wrong_flip += 1

        vcr = changed / max(total_with_tool, 1)
        cfr = correct_flip / max(changed, 1)

        result = {
            "n_with_tool":          total_with_tool,
            "n_changed":            changed,
            "verdict_change_rate":  round(vcr, 4),
            "correct_flip_n":       correct_flip,
            "wrong_flip_n":         wrong_flip,
            "correct_flip_rate":    round(cfr, 4),
        }
        self._agent_summary["verdict_change_rate"]  = round(vcr, 4)
        self._agent_summary["correct_flip_rate"]    = round(cfr, 4)
        return result

    # ── 3-4. Action Distribution ─────────────────────────────

    def calculate_action_distribution(
        self, predictions: List[Dict]
    ) -> Dict:
        """
        agent_trace에서 tool action 분포 집계.
        batch_inference_agent.py의 FOLLOWUP_ACTIONS 기준보다 정규화된 방식.
        """
        expand_n = reroute_n = accept_n = fast_n = total = 0
        for pred in predictions:
            if pred.get("error"):
                continue
            total += 1
            trace = pred.get("agent_trace", [])
            track = pred.get("track", "deep")
            if track == "fast":
                fast_n += 1
                continue
            has_expand  = any("expand_within_cluster"  in s for s in trace)
            has_reroute = any("reroute_cross_cluster"  in s for s in trace)
            if has_expand:
                expand_n += 1
            elif has_reroute:
                reroute_n += 1
            else:
                accept_n += 1

        deep_n = total - fast_n
        result = {
            "total":           total,
            "fast_track_n":    fast_n,
            "deep_track_n":    deep_n,
            "expand_n":        expand_n,
            "reroute_n":       reroute_n,
            "accept_direct_n": accept_n,
            "expand_rate":     round(expand_n / max(deep_n, 1), 4),
            "reroute_rate":    round(reroute_n / max(deep_n, 1), 4),
            "fast_rate":       round(fast_n / max(total, 1), 4),
        }
        self._agent_summary["expand_rate"]  = round(expand_n / max(deep_n, 1), 4)
        self._agent_summary["reroute_rate"] = round(reroute_n / max(deep_n, 1), 4)
        return result

    # ── 3-5. Agent Latency Breakdown ────────────────────────

    def calculate_agent_latency_breakdown(
        self, predictions: List[Dict]
    ) -> Dict:
        """
        agent loop의 구간별 latency 분해.
        base_retrieve: orchestrator base call (retriever)
        total:         전체 end-to-end
        tool_delta:    total - base_retrieve (analyzer + planner + tool 합산)
        """
        base_lats = []
        total_lats = []
        tool_deltas = []

        for pred in predictions:
            if pred.get("error"):
                continue
            lms = pred.get("latency_ms", {})
            if isinstance(lms, dict):
                base  = lms.get("base_retrieve")
                total = lms.get("total")
                if base is not None and total is not None:
                    base_lats.append(float(base))
                    total_lats.append(float(total))
                    tool_deltas.append(float(total) - float(base))

        def _stats(arr):
            if not arr:
                return {"mean": None, "p50": None, "p95": None}
            a = np.array(arr)
            return {
                "mean": round(float(np.mean(a)), 2),
                "p50":  round(float(np.percentile(a, 50)), 2),
                "p95":  round(float(np.percentile(a, 95)), 2),
                "_arr": arr,  # 그래프용 raw 데이터
            }

        result = {
            "base_retrieve_ms": _stats(base_lats),
            "total_ms":         _stats(total_lats),
            "tool_delta_ms":    _stats(tool_deltas),   # analyzer+planner+tool 합산
            "n_evaluated":      len(base_lats),
        }
        if total_lats:
            self._agent_summary["latency_total_p95_ms"] = round(float(np.percentile(total_lats, 95)), 2)
        return result

    # ── Summary ──────────────────────────────────────────────

    def get_agent_summary(self) -> Dict:
        return {**self.get_summary(), **self._agent_summary}

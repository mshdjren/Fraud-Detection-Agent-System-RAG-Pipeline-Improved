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
from collections import Counter, defaultdict
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
        b_total = nb_total = 0        # ★ [E3-a] in-loop 집계 — 아래 raw 재계산 제거

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
                b_total += 1               # ★ [E3-a]
                if label == 1:  # 실제 fraud
                    b_fraud += 1
                    if cls == "ABNORMAL":
                        b_correct += 1
                # expand/reroute 발동 여부
                # ★ [E3-b] 정확일치 → substring: 실행 엔트리는 "<tool>:<mode>" 형태라
                #   콜론 포함 prefix 매칭이면 충분하고, 배포 env 오탈자
                #   ("mcp_navtive" — 06-controller.yaml)로 생기는 미집계도 흡수한다.
                #   (plan 단계 "plan:<tool>"은 콜론 위치가 달라 매칭되지 않음 = 의도 유지)
                has_tool = any(("expand_within_cluster:" in s)
                               or ("reroute_cross_cluster:" in s) for s in trace)
                if has_tool:
                    b_expand += 1
                    if r0_cls and r0_cls != cls:
                        b_changed += 1
            else:
                nb_total += 1              # ★ [E3-a]
                if label == 1 and cls == "ABNORMAL":
                    nb_correct += 1
                if label == 1:
                    nb_fraud += 1

        # ★ [E3-a] b_total/nb_total은 loop에서 in_boundary와 동일 기준으로 집계됨
        #   (기존: band [0.3,0.55)를 raw L2(≈0.8~1.5)에 재적용 → boundary_n=0 무효)

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

    # ★ [P6] legacy tool(expand/reroute)과 학습 planner action을 같은 축에서
    #   본다. 기존 지표는 trace의 tool 이름만 봐서 PLANNER_BACKEND=local일 때
    #   재검색을 통째로 놓쳤다(실측: 재검색 15건인데 expand_n=reroute_n=0,
    #   verdict_change n_with_tool=0).
    _PLANNER_RESEARCH_ACTIONS = ("contextual_knn", "attribute_knn", "global_knn")
    # ★ [하위호환] controller 수정 전 로그는 게이트 사유를 planner_backend에
    #   섞어 넣고(local:empty_pool) invalid_reason으로 보냈다. 구버전 리포트도
    #   같은 축으로 읽히도록 여기서 정규화한다.
    _GATE_REASONS = ("empty_candidate_pool", "max_steps_reached",
                     "expansion_already_used", "out_of_training_population",
                     "no_executable_action")

    @classmethod
    def _research_tool_of(cls, pred: Dict) -> Optional[str]:
        """이 샘플이 실제로 실행한 재검색 tool 이름. 없으면 None."""
        _pa = pred.get("_planner_action")
        if _pa in cls._PLANNER_RESEARCH_ACTIONS:
            return _pa
        trace = pred.get("agent_trace", []) or []
        for _t in cls._PLANNER_RESEARCH_ACTIONS + ("expand_within_cluster",
                                                   "reroute_cross_cluster"):
            if any(str(s).startswith(_t) for s in trace):
                return _t
        # planner 트레이스가 없어도 controller finalize가 researched를 남긴다
        return "unknown_research" if pred.get("_researched") else None

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
            # ★ [P6] legacy tool + planner action 공통 판정
            if self._research_tool_of(pred) is None:
                continue
            total_with_tool += 1

            r0  = pred.get("_round0_classification")
            r1  = pred.get("classification")
            lbl = int(gt_labels[i])
            if r0 and r1 and r0 != r1:
                changed += 1
                # ★ [E4 수정] binary flip은 GT 정합을 반드시 바꾼다 —
                #   기존 코드는 lbl==1만 검사해 gt=0(정상)을 오탐으로 뒤집는
                #   wrong flip이 미집계 (실측: changed 4 = correct 2 + 미집계 2).
                _gt_cls = "ABNORMAL" if lbl == 1 else "NORMAL"
                if r1 == _gt_cls:
                    correct_flip += 1     # 틀린 판정 → 맞는 판정
                else:
                    wrong_flip += 1       # 맞는 판정 → 틀린 판정

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
        # ★ [P6] planner action별 카운터 — legacy 키는 그대로 두고 추가한다
        planner_n = Counter()
        for pred in predictions:
            if pred.get("error"):
                continue
            total += 1
            trace = pred.get("agent_trace", [])
            track = pred.get("track", "deep")
            if track == "fast":
                fast_n += 1
                continue
            _rt = self._research_tool_of(pred)
            if _rt:
                planner_n[_rt] += 1
            if _rt == "expand_within_cluster":
                expand_n += 1
            elif _rt == "reroute_cross_cluster":
                reroute_n += 1
            elif _rt is None:
                accept_n += 1
            else:
                # planner 계열 재검색 — accept로 세면 "재검색 0건"으로 보인다
                pass

        # ★ [direction 비율] agent_rounds의 what_missing prefix로
        #   cluster_mismatch(cm) / neighbor_insufficient(ni) / 무출력(none) 집계 —
        #   "expand rate"만으로는 evaluator가 어떤 결함 유형을 보고했는지 안 보임.
        cm_n = ni_n = wm_none_n = wm_rounds = 0
        for pred in predictions:
            if pred.get("error"):
                continue
            for r in (pred.get("agent_rounds") or []):
                if not isinstance(r.get("round"), int):
                    continue                      # "final" 요약 엔트리 제외
                if str(r.get("sufficiency", "")).lower() != "ambiguous":
                    continue                      # wm 정답이 null인 confident 제외
                wm_rounds += 1
                wm = r.get("what_missing")
                if not isinstance(wm, str) or not wm.strip():
                    wm_none_n += 1
                elif wm.startswith("cluster_mismatch"):
                    cm_n += 1
                elif wm.startswith("neighbor_insufficient"):
                    ni_n += 1
                else:
                    wm_none_n += 1                # 형식 이탈도 none으로 집계

        deep_n = total - fast_n
        result = {
            "total":           total,
            "fast_track_n":    fast_n,
            "deep_track_n":    deep_n,
            "expand_n":        expand_n,
            "reroute_n":       reroute_n,
            # ★ [P6] planner action 집계 (contextual/attribute/global)
            "planner_action_n":    dict(planner_n),
            "research_n":          int(sum(planner_n.values())),
            "research_rate":       round(sum(planner_n.values()) / max(total, 1), 4),
            "accept_direct_n": accept_n,
            "expand_rate":     round(expand_n / max(deep_n, 1), 4),
            "reroute_rate":    round(reroute_n / max(deep_n, 1), 4),
            "fast_rate":       round(fast_n / max(total, 1), 4),
            # direction 분포 (ambiguous 라운드 기준)
            "wm_rounds_n":     wm_rounds,
            "cm_n":            cm_n,
            "ni_n":            ni_n,
            "wm_none_n":       wm_none_n,
            "cm_ratio":        round(cm_n / max(wm_rounds, 1), 4),
            "ni_ratio":        round(ni_n / max(wm_rounds, 1), 4),
        }
        self._agent_summary["expand_rate"]  = round(expand_n / max(deep_n, 1), 4)
        self._agent_summary["reroute_rate"] = round(reroute_n / max(deep_n, 1), 4)
        self._agent_summary["cm_ratio"]     = result["cm_ratio"]
        self._agent_summary["ni_ratio"]     = result["ni_ratio"]
        return result

    # ============================================================
    # ★ [V3] 검색 품질 지표 — verdict/AUROC와 독립인 evidence-completing 축
    # ============================================================
    def compute_retrieval_quality(self, predictions: list) -> dict:
        """v3 트레이스(_nd_global/_flip/_wm_type/_contrast_*)로 검색 품질 요약.
        legacy 실행(필드 전부 None)에서는 counts=0인 무해한 dict 반환."""
        import numpy as _np
        nd0, ndg, flips, band_n = [], [], 0, 0
        wm_types, contrast_n, contrast_ratio = {}, 0, []
        flip_normal = flip_total = 0
        for p in predictions:
            g = p.get("_nd_global")
            b = p.get("_distance_band")
            wt = p.get("_wm_type")
            if wt:
                wm_types[wt] = wm_types.get(wt, 0) + 1
            if b == "boundary":
                band_n += 1
                if p.get("_flip"):
                    flip_total += 1
                    if str(p.get("gt_label", p.get("label", ""))).upper() in ("0", "NORMAL"):
                        flip_normal += 1
            if g is not None:
                _n0 = p.get("_nd_round0") or p.get("_nd_final")
                if _n0:
                    nd0.append(float(_n0)); ndg.append(float(g))
            cd, t1 = p.get("_contrast_distance"), p.get("top_1_distance")
            if cd is not None:
                contrast_n += 1
                if t1:
                    contrast_ratio.append(float(cd) / max(float(t1), 1e-9))
        eps = ([max(a / max(b_, 1e-9) - 1.0, 0.0) for a, b_ in zip(nd0, ndg)]
               if ndg else [])
        rq = {
            "verify_global_n":   len(ndg),
            "epsilon_median":    round(float(_np.median(eps)), 4) if eps else None,
            "epsilon_p90":       round(float(_np.percentile(eps, 90)), 4) if eps else None,
            "flip_rate_boundary": round(flip_total / max(band_n, 1), 4),
            "flip_normal_purity": round(flip_normal / max(flip_total, 1), 4) if flip_total else None,
            "wm_type_dist":      wm_types,
            "contrast_usage_n":  contrast_n,
            "contrast_dist_ratio_median": (round(float(_np.median(contrast_ratio)), 4)
                                           if contrast_ratio else None),
        }
        self._agent_summary["retrieval_quality"] = rq
        return rq

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
        fwd_lats = []          # ★ forward 모드: analyzer 1회 호출 구간
        ctrl_lats = []         # agent 모드: controller loop 전체 구간

        for pred in predictions:
            if pred.get("error"):
                continue
            lms = pred.get("latency_ms", {})
            if isinstance(lms, dict):
                base  = lms.get("base_retrieve")
                total = lms.get("total")
                if lms.get("forward_analyzer") is not None:
                    fwd_lats.append(float(lms["forward_analyzer"]))
                if lms.get("controller_agent_loop") is not None:
                    ctrl_lats.append(float(lms["controller_agent_loop"]))
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
        if fwd_lats:                                   # ★ forward 모드 구간
            result["forward_analyzer_ms"] = _stats(fwd_lats)
        if ctrl_lats:                                  # agent 모드 loop 구간
            result["controller_loop_ms"] = _stats(ctrl_lats)
        if total_lats:
            self._agent_summary["latency_total_p95_ms"] = round(float(np.percentile(total_lats, 95)), 2)
        return result

    # ── Summary ──────────────────────────────────────────────

    def get_agent_summary(self) -> Dict:
        return {**self.get_summary(), **self._agent_summary}

    # ============================================================
    # ★ [Planner] 재검색 정책 / 증거 신규성 / 예산 지표
    # ============================================================
    # 오라클 M0(EVPI·rescued/degraded·동률)가 "상한이 얼마인가"를 오프라인에서
    # 답했다면, 아래 셋은 "서빙에서 실제로 얼마를 회수했는가"를 답한다.
    # research_effect(nd 변화 인과)와 축이 다르다:
    #   planner_policy  — planner가 무엇을 골랐는가 (행동)
    #   evidence_novelty— 실제로 다른 증거를 가져왔는가 (수단)
    #   budget_curve    — 그 비용이 값어치를 했는가 (경제성)

    @staticmethod
    def _valid_rows(predictions: List[Dict], gt_labels: np.ndarray):
        out = []
        for i, p in enumerate(predictions):
            if p.get("error") or i >= len(gt_labels):
                continue
            out.append((p, int(gt_labels[i])))
        return out

    def calculate_planner_policy_metrics(
        self, predictions: List[Dict], gt_labels: np.ndarray,
        cls_threshold: float = 0.5,
    ) -> Dict:
        """planner의 행동 분포와 포맷 건전성.

        ★ 핵심 확인 사항 두 가지
          1) search_rate — 학습에서 관측된 '전부 검색' 편향이 서빙에서도
             재현되는가. oracle stop 비율(~86%)과 나란히 봐야 의미가 있다.
          2) fallback/invalid rate — 제약 디코딩(GBNF/trie)이 서빙에서도
             실제로 작동하는가. 학습은 trie, 서빙은 GBNF라 토큰화 경로가
             달라 여기서만 드러나는 불일치가 있을 수 있다. 0이 아니면
             train-serve 디코딩 gap을 의심해야 한다.
        """
        rows = self._valid_rows(predictions, gt_labels)
        if not rows:
            return {"n": 0}

        acts, sources, invalids, gates, stages, backends = [], [], [], [], set(), set()
        lat, ntok, costs = [], [], []
        n_fallback = n_researched = 0
        for p, _ in rows:
            a = p.get("_planner_action")
            acts.append(a if a else "(no_planner)")
            if p.get("_planner_source"):
                sources.append(p["_planner_source"])
            # ★ [P5] 정상 게이트(empty_pool/max_steps)와 진짜 invalid(파싱 실패,
            #   stage 밖 action)를 분리한다. 섞으면 "invalid 5건"으로 보여
            #   제약 디코딩이 깨진 것처럼 오독된다 — 실제로는 정책대로 stop.
            _gr = p.get("_planner_gate_reason")
            _ir = p.get("_planner_invalid_reason")
            if not _gr and _ir and str(_ir).split(":")[0].split("(")[0] in self._GATE_REASONS:
                _gr = _ir          # 구버전 로그 호환
                _ir = None
            if _gr:
                gates.append(str(_gr).split("(")[0].split(":")[0])
            elif _ir:
                invalids.append(str(_ir).split(":")[0])
            if p.get("_planner_is_fallback"):
                n_fallback += 1
            if p.get("_researched"):
                n_researched += 1
            for _v, _acc in ((p.get("_planner_latency_ms"), lat),
                             (p.get("_planner_n_out_tokens"), ntok),
                             (p.get("_planner_cost"), costs)):
                if _v is not None:
                    _acc.append(float(_v))
            if p.get("_planner_stage"):
                stages.add(p["_planner_stage"])
            if p.get("_planner_backend"):
                # ★ [P5] 게이트 사유가 섞여 들어오던 접두사(local:empty_pool 등)를
                #   제거해 backend 축을 순수하게 유지한다(구버전 로그 호환).
                _bk = str(p["_planner_backend"])
                if _bk.startswith("local:"):
                    _sfx = _bk.split(":", 1)[1]
                    # 게이트 마커는 backend가 아니다 — 구버전 로그에서 걸러낸다
                    _bk = None if (_sfx in ("empty_pool",) or "guard" in _sfx) else _sfx
                # ★ 게이트 행은 backend 축에 넣지 않는다. 모델을 안 태운
                #   행까지 세면 backends가 늘 2개로 보여 오탐 경고가 뜬다.
                if _bk and _bk != "local":
                    backends.add(_bk)

        n = len(rows)
        n_planner = sum(1 for a in acts if a != "(no_planner)")
        dist = {k: round(v / n, 4) for k, v in Counter(acts).items()}
        # ★ [P5] 모델이 직접 고른 행만 — 게이트 stop을 섞으면 정책이 왜곡된다
        _model_acts = [a for a, (p, _) in zip(acts, rows)
                       if p.get("_planner_source") == "model"]
        _model_dist = ({k: round(v / len(_model_acts), 4)
                        for k, v in Counter(_model_acts).items()}
                       if _model_acts else {})

        def _pct(v, q):
            return round(float(np.percentile(v, q)), 1) if v else None

        out = {
            "n": n,
            "n_with_planner_trace": n_planner,
            "action_distribution": dist,
            "search_rate": round(n_researched / n, 4),
            "stop_rate": round(1 - n_researched / n, 4),
            "fallback_rate": (round(n_fallback / n_planner, 4) if n_planner else None),
            # invalid = 모델 출력이 잘못된 경우 (0이어야 정상 — 제약 디코딩 검증)
            "invalid_reasons": dict(Counter(invalids)),
            # gate = 정책대로 LLM 없이 stop한 경우 (정상 동작, invalid 아님)
            "gate_reasons": dict(Counter(gates)),
            "n_gated": len(gates),
            "gate_rate": round(len(gates) / n, 4),
            # 모델이 실제로 고른 것만 — 게이트를 빼야 정책 분포가 보인다
            "model_action_distribution": _model_dist,
            "source_distribution": dict(Counter(sources)),
            "planner_latency_ms": {"p50": _pct(lat, 50), "p95": _pct(lat, 95)},
            "output_tokens": {"p50": _pct(ntok, 50), "max": (max(ntok) if ntok else None)},
            "mean_cost": (round(float(np.mean(costs)), 4) if costs else None),
            "stages_seen": sorted(stages),
            "backends_seen": sorted(backends),
        }
        # ★ 설정이 샘플마다 다르면 리포트 전체가 혼합 데이터다 — 즉시 경고.
        if len(stages) > 1 or len(backends) > 1:
            out["_warning"] = ("planner stage/backend가 run 내에서 일관되지 않음 — "
                               f"stages={sorted(stages)} backends={sorted(backends)}")
        # ★ 모델이 stop을 한 번도 고르지 않으면 게이팅을 학습하지 못한 것이다.
        #   (게이트로 인한 stop은 정책이지 모델의 결정이 아니므로 분리해서 본다)
        _mad = out.get("model_action_distribution") or {}
        if _mad and "stop" not in _mad:
            out["_warning_no_stop"] = (
                "모델이 stop을 한 번도 선택하지 않음 — '언제 검색할지'가 학습되지 "
                "않았거나 base prior가 stop에 질량을 주지 않는다")
        if n_planner and n_fallback / n_planner > 0.05:
            out["_warning_fallback"] = (
                "fallback 5% 초과 — 제약 디코딩 미적용 또는 candidate_pool 불일치 의심")
        self._agent_summary["planner_search_rate"] = out["search_rate"]
        self._agent_summary["planner_fallback_rate"] = out["fallback_rate"]
        return out

    def calculate_evidence_novelty(
        self, predictions: List[Dict], gt_labels: np.ndarray,
        cls_threshold: float = 0.5,
    ) -> Dict:
        """재검색이 '다른 증거'를 가져왔는지 — nd 개선과 분리해서 측정.

        비판적 검토 §3의 요구사항: "nd가 좋아져서 좋은 retrieval인가"와
        "실제로 다른 evidence를 가져왔고 그것이 decision에 기여했는가"를
        분리 기록할 것. jaccard가 1.0에 가까우면 재검색이 이웃을 거의
        바꾸지 못한 것이고, 그 상태에서 nd만 움직였다면 그것은 새 증거가
        아니라 척도 변화다.

        attribute_knn 도입 논거(dense 축이 못 보는 후보 복구)는 여기서만
        검증된다 — action별 novelty가 갈리지 않으면 두 tool은 같은 도구다.
        """
        rows = self._valid_rows(predictions, gt_labels)
        res = [(p, y) for p, y in rows if p.get("_researched")]
        if not res:
            return {"n_researched": 0,
                    "_note": "재검색된 샘플이 없어 novelty를 정의할 수 없음"}

        by_action: Dict[str, List[float]] = defaultdict(list)
        jac, nov, repl, esc, sup = [], [], [], [], []
        for p, _ in res:
            j = p.get("_neighbor_jaccard")
            if j is not None:
                jac.append(float(j))
                by_action[p.get("_planner_action") or "unknown"].append(float(j))
            if p.get("_neighbor_novelty") is not None:
                nov.append(float(p["_neighbor_novelty"]))
            if p.get("_n_neighbors_replaced") is not None:
                repl.append(float(p["_n_neighbors_replaced"]))
            if p.get("_cross_cluster_escape") is not None:
                esc.append(float(p["_cross_cluster_escape"]))
            if p.get("_research_support") is not None:
                sup.append(float(p["_research_support"]))

        def _m(v):
            return round(float(np.mean(v)), 4) if v else None

        out = {
            "n_researched": len(res),
            "n_with_neighbor_trace": len(jac),
            "mean_jaccard_round0_final": _m(jac),
            "mean_novelty": _m(nov),
            "mean_neighbors_replaced": _m(repl),
            "cross_cluster_escape_rate": _m(esc),
            "mean_support": _m(sup),
            "jaccard_by_action": {k: round(float(np.mean(v)), 4)
                                  for k, v in by_action.items() if v},
            "n_by_action": {k: len(v) for k, v in by_action.items()},
        }
        if len(out["jaccard_by_action"]) >= 2:
            _vals = list(out["jaccard_by_action"].values())
            out["jaccard_spread_across_actions"] = round(max(_vals) - min(_vals), 4)
            out["_interpretation"] = (
                "action별 jaccard 차이가 작으면 두 tool이 같은 이웃을 가져오는 것 — "
                "이질 축 도입 논거가 성립하지 않는다")
        if out["mean_jaccard_round0_final"] is not None and \
                out["mean_jaccard_round0_final"] > 0.9:
            out["_warning"] = ("jaccard>0.9 — 재검색이 이웃을 거의 바꾸지 못했다. "
                               "nd만 움직였다면 새 증거가 아니라 척도 변화다")
        self._agent_summary["evidence_novelty"] = out["mean_novelty"]
        return out

    def calculate_budget_curve(
        self, predictions: List[Dict], gt_labels: np.ndarray,
        cls_threshold: float = 0.5,
    ) -> Dict:
        """검색 예산 B% 하에서의 정확도/지연 — '검색이 값어치를 했는가'.

        ★ 노트북 [M4] 예산 곡선의 서빙 대응물이다. 다만 여기서는 reward가
        아니라 **실제 판정 정확도**로 잰다(서빙에는 오라클 lookup이 없다).
        Δnd가 큰 순으로 정렬해 상위 B%만 재검색했다고 가정하고 재계산한다 —
        이미 실행된 결과를 사후 절단하는 것이라 추가 호출이 없다.

        B=0%가 최적이면 재검색은 순손실이고, 곡선이 볼록하면 게이팅으로
        회수 가능하다는 뜻이다.
        """
        rows = self._valid_rows(predictions, gt_labels)
        items = []
        for p, y in rows:
            nd0, ndf = p.get("_nd_round0"), p.get("_nd_final")
            if nd0 is None:
                continue
            nd0 = float(nd0)
            ndf = float(ndf) if ndf is not None else nd0
            items.append({
                "y": y, "nd0": nd0, "ndf": ndf,
                "researched": bool(p.get("_researched")),
                "gain": abs(ndf - nd0),
                "research_ms": float(p.get("_research_ms") or 0.0),
            })
        if not items:
            return {"n": 0, "_note": "nd_round0 트레이스 없음 — controller finalize 확인"}

        y = np.array([it["y"] for it in items])
        nd0 = np.array([it["nd0"] for it in items])
        ndf = np.array([it["ndf"] for it in items])
        gain = np.array([it["gain"] for it in items])
        ms = np.array([it["research_ms"] for it in items])
        base_acc = float(((nd0 >= cls_threshold).astype(int) == y).mean())

        order = np.argsort(-gain)          # Δnd 큰 순
        curve = []
        for b in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0):
            k = int(round(b * len(items)))
            use = np.zeros(len(items), dtype=bool)
            use[order[:k]] = True
            nd_mix = np.where(use, ndf, nd0)
            acc = float(((nd_mix >= cls_threshold).astype(int) == y).mean())
            curve.append({"budget": b, "n_search": k,
                          "accuracy": round(acc, 4),
                          "delta_vs_no_search": round(acc - base_acc, 4),
                          "added_latency_ms_total": round(float(ms[use].sum()), 1)})

        best = max(curve, key=lambda c: c["accuracy"])
        out = {
            "n": len(items),
            "cls_threshold": cls_threshold,
            "accuracy_no_search": round(base_acc, 4),
            "curve": curve,
            "best_budget": best["budget"],
            "best_accuracy": best["accuracy"],
            "gain_at_best": best["delta_vs_no_search"],
        }
        if best["budget"] == 0.0:
            out["_interpretation"] = (
                "B=0%가 최적 — 현재 정책에서 재검색은 순손실. 게이팅(언제 검색할지)이 "
                "학습되지 않았거나 tool 자체에 이득이 없다")
        elif best["budget"] < 1.0:
            out["_interpretation"] = (
                f"B={best['budget']:.0%}에서 최적 — 전부 검색하는 것보다 선택적 검색이 "
                f"낫다. 게이팅으로 {best['delta_vs_no_search']:+.4f} 회수 가능")
        self._agent_summary["budget_best"] = best["budget"]
        self._agent_summary["budget_best_gain"] = best["delta_vs_no_search"]
        return out
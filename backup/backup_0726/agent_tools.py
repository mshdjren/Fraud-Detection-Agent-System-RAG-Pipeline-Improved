"""
agent_tools.py — Agent workflow tool layer (simplified, initial stage)
=====================================================================
초기 단계 agent workflow에 필요한 최소 tool 계층만 유지한다.

구성:
  1) Deterministic pre-router 결정 (fast / deep)           — LLM 개입 없음
  2) Follow-up tool 카탈로그 (planner가 description 보고 1개 선택)
        - expand_within_cluster : 같은 클러스터, 이웃 확대   (evidence 양 부족)
        - reroute_cross_cluster : 다른 클러스터로 재라우팅   (클러스터/위치 의심)
        - compare_features      : (옵션, Arm 3) feature 대조 (이상 근거 설명 부족)
        - accept_verdict        : 충분 → 종료
  3) compute_feature_comparison — compare_features 실행 helper

제거된 부가 요소(초기 단계에서 불필요):
  view dispatch(persona/evidence/dual), S1·S2 band, PyOD placeholder,
  CostTracker, prefix-cache hint, AgentTool/choose_tool.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Literal


# ============================================================
# 1) Deterministic Pre-Router (fast / deep)
# ============================================================
# 직전 논의: fast/deep 선택은 global L2 distance 하나로 끝나는 수치 결정 →
# LLM이 reasoning할 의미적 재료가 없으므로 결정론적 pre-router로 분리한다.

def decide_track(
    global_distance: float,
    *,
    fast_threshold: float,
) -> Literal["fast", "deep"]:
    """
    global centroid 거리가 fast_threshold 미만이면 fast(즉시 종료),
    그 외에는 deep(검색 → 분석 → planner 루프).
    """
    return "fast" if float(global_distance) < float(fast_threshold) else "deep"


# ============================================================
# 2) Follow-up Tool Catalog — planner가 reasoning으로 1개 선택
# ============================================================
# 이질화 축 = 반환 신호의 종류 (양 / 위치 / 설명).
# use_when 경계는 Anthropic ACI 가이드(비슷한 tool이 여럿이면 경계 정의가
# 특히 중요)를 따라 명시. planner는 이 description만 보고 선택한다.

# Arm 3 토글: True면 compare_features를 planner 선택지에 포함.
ENABLE_COMPARE_FEATURES = os.getenv("ENABLE_COMPARE_FEATURES", "false").lower() == "true"

_ACCEPT_TOOL: Dict[str, str] = {
    "name":     "accept_verdict",
    "returns":  "—",
    "use_when": "analyzer reasoning이 명확·자기일관적이고 추가 증거가 verdict를 "
                "바꿀 가능성이 낮음. (재검색 비용을 쓰지 않고 종료)",
}
_EXPAND_TOOL: Dict[str, str] = {
    "name":     "expand_within_cluster",
    "returns":  "같은 클러스터의 정상 이웃을 더 많이 (더 깊은 evidence)",
    "use_when": "reasoning이 '이웃이 너무 적다 / 정상 범위를 단정 못 한다'처럼 "
                "evidence 양 부족을 호소. 클러스터 매칭 자체는 신뢰됨.",
}
_REROUTE_TOOL: Dict[str, str] = {
    "name":     "reroute_cross_cluster",
    "returns":  "다른 후보 클러스터들의 정상 이웃 (다른 공간으로 이동)",
    "use_when": "reasoning이 '이 샘플이 매칭된 persona/클러스터와 맞지 않는다'처럼 "
                "라우팅 불일치를 시사. 매칭된 클러스터 자체가 의심될 때.",
}
_COMPARE_TOOL: Dict[str, str] = {
    "name":     "compare_features",
    "returns":  "test ↔ 최근접 정상 간 feature 단위 편차 (이상 '근거' 설명)",
    "use_when": "이웃 양은 충분하나 reasoning이 '왜 이상인지' 구체적 근거를 "
                "짚지 못함. 검색을 늘리는 대신 가진 이웃과의 대조가 필요할 때.",
}

FOLLOWUP_TOOLS: List[Dict[str, str]] = [_ACCEPT_TOOL, _EXPAND_TOOL, _REROUTE_TOOL]
if ENABLE_COMPARE_FEATURES:
    FOLLOWUP_TOOLS.append(_COMPARE_TOOL)

# controller(planner)와 orchestrator(system)가 공유하는 단일 출처
FOLLOWUP_ACTIONS = tuple(t["name"] for t in FOLLOWUP_TOOLS)


def build_followup_tool_catalog() -> str:
    """planner 프롬프트에 주입할 tool 카탈로그 블록 (description 기반 선택용)."""
    lines = ["## AVAILABLE FOLLOW-UP TOOLS (select EXACTLY ONE)"]
    for t in FOLLOWUP_TOOLS:
        lines.append(f"- {t['name']}")
        lines.append(f"    returns:  {t['returns']}")
        lines.append(f"    use_when: {t['use_when']}")
    return "\n".join(lines)


# ============================================================
# 3) compare_features 계산 — test ↔ 최근접 정상 feature 대조
# ============================================================

def compute_feature_comparison(
    test_transaction: Dict[str, Any],
    nearest_normal: Optional[Dict[str, Any]] = None,
    *,
    numeric_keys=("purchase_value", "age"),
    top_n: int = 3,
) -> Dict[str, Any]:
    """
    test와 최근접 정상 이웃의 원본 feature를 feature 단위로 차분해
    가장 크게 벗어난 feature를 산출 → 이상의 '근거' hint 생성.

    nearest_normal이 None(이웃 원본 feature 미제공)이면 z-score 크기 기준
    deviation(정상 평균=0 으로부터의 편차)으로 graceful degrade.
    (※ retriever가 top-1 이웃 원본 feature를 응답에 포함하면 완전한 대조로 승격.)

    Returns: {"deltas": [{"feature","test","normal","delta","abs"}...], "hint": str}
    """
    deltas: List[Dict[str, Any]] = []
    for k in numeric_keys:
        tv = test_transaction.get(k)
        if tv is None:
            continue
        try:
            tvf = float(tv)
        except (TypeError, ValueError):
            continue

        nvf: Optional[float] = None
        if nearest_normal is not None and nearest_normal.get(k) is not None:
            try:
                nvf = float(nearest_normal.get(k))
            except (TypeError, ValueError):
                nvf = None

        d = tvf - nvf if nvf is not None else tvf  # degrade: z-score 편차
        deltas.append({
            "feature": k, "test": tvf, "normal": nvf,
            "delta": round(d, 4), "abs": abs(d),
        })

    deltas.sort(key=lambda x: x["abs"], reverse=True)
    top = deltas[:top_n]
    if top:
        parts = []
        for d in top:
            if d["normal"] is None:
                parts.append(f"{d['feature']} deviates z={d['delta']:+.2f} from normal mean")
            else:
                parts.append(f"{d['feature']} differs {d['delta']:+.2f} vs nearest normal")
        hint = "Feature comparison — " + "; ".join(parts) + ". Ground the verdict on these features."
    else:
        hint = "No numeric feature deviation computable."
    return {"deltas": deltas, "hint": hint}

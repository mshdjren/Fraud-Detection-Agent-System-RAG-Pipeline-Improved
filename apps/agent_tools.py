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

# ★ [V3] 재검색 의미론 스위치 — "legacy"(reroute/expand) | "v3"(verify_global/
#   expand_reweight + fetch_contrast). tool 이름(enum)은 유지하고 description만
#   교체해 controller 그래프/FOLLOWUP_ACTIONS 무수정으로 A/B 가능하게 한다.
RESEARCH_MODE = os.getenv("RESEARCH_MODE", "legacy").lower()

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

# ── [V3] 의미 교체판 description (이름 동일 — enum/그래프 하위호환) ──────
_EXPAND_TOOL_V3: Dict[str, str] = {
    "name":     "expand_within_cluster",
    "returns":  "같은 클러스터 top-k 확대 → PatchCore reweighting으로 nd 재산출 "
                "(판정 변경이 아니라 추정 분산 축소)",
    "use_when": "what_missing=neighbor_insufficient: dominant persona 축은 일치하나 이웃 거리 spread가 커서 "
                "nd 추정이 불안정할 때. (aka expand_reweight)",
}
_REROUTE_TOOL_V3: Dict[str, str] = {
    "name":     "reroute_cross_cluster",
    "returns":  "전역(무필터) HNSW top-k — cluster 제한이 만든 거리 인플레이션(ε) 교정, "
                "nd의 진짜 하한 + 실제 귀속 cluster",
    "use_when": "what_missing=cluster_mismatch: dominant persona 축에서 이탈 → cluster 제한이 증거를 가릴 때 "
                "전역 1회로 검증. target 추측 불필요. (aka verify_global)",
}
_CONTRAST_TOOL: Dict[str, str] = {
    "name":     "fetch_contrast",
    "returns":  "불일치 축에서 test와 같은 값을 가진 클러스터 내 정상 top1 "
                "(감사 주석 인용용 대조 증거 — 판정 불개입)",
    "use_when": "wwhat_missingm=neighbor_insufficient 보조: '같은 프로필의 정상은 이랬다'를 근거로 인용해야 할 때.",
}

if RESEARCH_MODE == "v3":
    FOLLOWUP_TOOLS: List[Dict[str, str]] = [_ACCEPT_TOOL, _EXPAND_TOOL_V3, _REROUTE_TOOL_V3]
else:
    FOLLOWUP_TOOLS = [_ACCEPT_TOOL, _EXPAND_TOOL, _REROUTE_TOOL]
if ENABLE_COMPARE_FEATURES:
    FOLLOWUP_TOOLS.append(_COMPARE_TOOL)

# fetch_contrast는 planner 선택지가 아니라 ni 보조 호출 — MCP 노출용 메타만 유지
EXTRA_TOOLS: List[Dict[str, str]] = [_CONTRAST_TOOL] if RESEARCH_MODE == "v3" else []
EXTRA_TOOL_NAMES = tuple(t["name"] for t in EXTRA_TOOLS)

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


# ============================================================
# 4) GRPO Planner — action space (신규, 인수인계 §1 + 비판적 검토 §1/§2 반영)
# ============================================================
# 위의 FOLLOWUP_TOOLS(accept/expand/reroute, legacy·v3 공용)는 현재 서빙 중인
# Gemini/deterministic planner 전용이며 이 섹션은 건드리지 않는다. 아래는
# GRPO planner(0.5B, 아직 미학습 — 이 섹션은 오라클 rollout 준비용)의 별도
# action space다. 두 planner는 서로 다른 controller 경로(node_plan vs
# 신규 /rollout)를 타므로 카탈로그를 공유하지 않는다.
#
# 비판적 검토 §1 수용: retrieve(view, scope, features) 중 scope는 V1에서
#   제거하고 global로 고정한다. cluster-scoped contextual은 Stage 2 대상.
# 비판적 검토 §2 수용: features ⊆ agrees∪persona_match는 "정답 공간"이
#   아니라 "이 rollout에서 실행해볼 후보 생성 규칙"이다. 학습된 policy의
#   serving-time action space를 이 함수가 강제하지 않는다 — 지금은 오직
#   오라클 계산을 위해 유한하고 다루기 쉬운 후보 집합을 만드는 용도.

GRPO_ACTIONS: tuple = ("stop", "global_knn", "contextual_knn", "attribute_knn")

# retriever가 실제로 이웃 feature로 반환하는 컬럼(retriever.py _FEATURE_COLS)
# 중 categorical(soft should-boost 대상)만 후보로 삼는다. purchase_value/age는
# 연속값이라 term boost 대상이 아니다.
CONTEXTUAL_CANDIDATE_FIELDS: tuple = (
    "sex", "source", "browser", "IP_country", "month_purchase", "weekday_purchase",
)

MAX_CONTEXTUAL_FEATURES = int(os.getenv("MAX_CONTEXTUAL_FEATURES", "2"))

# Reward의 cost 항 (인수인계 §1 Reward Stage 1). scope를 global로 고정했으므로
# "contextual@cluster" variant는 존재하지 않는다 — Stage 2에서 scope를
# 재도입하면 그때 추가한다.
# ★ [cost 재설계] binary cost — 재검색=1, 종료=0.
#   기존 1.2/1.5/1.0은 실측 근거가 없는 값이었고 두 가지 문제를 만들었다:
#     (a) λ의 의미가 흐려진다. binary면 λ가 곧 "재검색 1회의 가격"이 되어
#         λ×balancing CAP 상호작용(handover §5 #12)의 절반이 해소된다.
#     (b) global_knn이 지는 이유가 cost(1.5) 때문인지 Δmargin 때문인지 분리되지
#         않았다. cost를 동일하게 두면 순수 Δmargin 비교가 된다.
#   ※ 기존 실험값 환산: 이전 λ=0.02 x cost 1.2 = 0.024 가 새 스케일의 등가 λ.
#   ※ 실측 latency 기반 재보정이 필요하면 COST_* env로 되돌릴 수 있고,
#     오라클에 delta_margin/cost 원본이 저장되므로 사후 재계산도 가능하다.
COST_RESEARCH = float(os.getenv("COST_RESEARCH", "1.0"))

ACTION_COST: Dict[str, float] = {
    "stop":           0.0,
    "contextual_knn": float(os.getenv("COST_CONTEXTUAL_KNN", str(COST_RESEARCH))),
    "global_knn":     float(os.getenv("COST_GLOBAL_KNN",     str(COST_RESEARCH))),
    "attribute_knn":  float(os.getenv("COST_ATTRIBUTE_KNN",  str(COST_RESEARCH))),
}


def candidate_feature_sets(
    agree_like_axes: List[str],
    *,
    max_features: int = MAX_CONTEXTUAL_FEATURES,
) -> List[List[str]]:
    """
    agree_like_axes(= agrees ∪ persona_match 후보 pool, 최대 CONTEXTUAL_
    CANDIDATE_FIELDS로 제한된 categorical 축 목록)에서 |f|<=max_features인
    비어있지 않은 부분집합을 전부 만든다. 이것이 오라클 rollout이 실제로
    실행해볼 contextual_knn 후보(핸드오버 §1 "action space ≈ 10~12개")다.

    ★ 이 함수는 "정답만 여기 있다"는 제약이 아니라 "오라클 계산을 다루기
    쉽게 만드는 후보 생성 규칙"이다(비판적 검토 §2). agree_like_axes 자체가
    틀렸을 가능성(evaluator 오류)을 이 함수가 해소해주지는 않는다 — 그건
    evaluator의 agrees 필드 정확도 문제이지 이 함수의 책임이 아니다.
    """
    pool = [f for f in dict.fromkeys(agree_like_axes) if f in CONTEXTUAL_CANDIDATE_FIELDS]
    if not pool:
        return []
    out: List[List[str]] = []
    import itertools
    for r in range(1, max(1, min(max_features, len(pool))) + 1):
        out.extend(list(c) for c in itertools.combinations(pool, r))
    return out


def build_grpo_action_space_v1(agree_like_axes: List[str]) -> List[Dict[str, Any]]:
    """
    state 하나에 대한 전수 action 목록: [{"action":"stop"}, {"action":
    "global_knn"}, {"action":"contextual_knn","features":[...]}, ...].
    오라클 rollout(controller.py /rollout)이 그대로 순회하며 각 action을
    전부 실행한다 — 샘플링이 아니라 전수 전개(인수인계 §5 "GRPO group
    다양성").
    """
    actions: List[Dict[str, Any]] = [{"action": "stop"}, {"action": "global_knn"}]
    for feats in candidate_feature_sets(agree_like_axes):
        actions.append({"action": "contextual_knn", "features": feats})
        actions.append({"action": "attribute_knn",  "features": feats})
    return actions

# ============================================================
# 5) Planner Stage Registry — action space / cost / tool 매핑의 단일 출처
# ============================================================
# stage가 바뀔 때 실제로 변하는 축은 4개다:
#   (a) action 종류   (b) scope 유무   (c) cost 키 스키마   (d) feature 상한
# 이 4개를 여기 한 곳에 모으고, controller(서빙/오라클) / local_planner(추론) /
# agent_tools_mcp_server(실행) / oracle_extractor(채집)가 전부 이 registry만
# 참조하게 한다 — stage 확장 시 수정 지점이 이 블록 하나로 고정된다.

PLANNER_STAGE = os.getenv("PLANNER_STAGE", "stage1").lower()

COST_CONTEXTUAL_KNN_CLUSTER = float(
    os.getenv("COST_CONTEXTUAL_KNN_CLUSTER", str(COST_RESEARCH)))

# requires_features 의미 — contextual과 attribute가 **동일한 후보 생성 규칙**을
# 쓴다(agrees∪persona_match, 최대 2개). 그래야 soft(rerank) vs hard(term filter)
# 비교가 같은 후보 집합 위에서 이루어져 깨끗하다.
_PLANNER_ACTION_SPECS: Dict[str, Dict[str, Any]] = {
    "stop": dict(
        action="stop", scope=None, tool_name=None, cost=ACTION_COST["stop"],
        requires_features=False, max_features=0,
        purpose="종료 — 추가 증거 없이 현재 nd로 판정"),
    "global_knn": dict(
        action="global_knn", scope="global", tool_name="global_knn",
        cost=ACTION_COST["global_knn"], requires_features=False, max_features=0,
        purpose="cluster 제한 해제 — round0의 superset. nd의 진짜 하한 확인"),
    "contextual_knn": dict(
        action="contextual_knn", scope="global", tool_name="contextual_knn",
        cost=ACTION_COST["contextual_knn"], requires_features=True,
        max_features=MAX_CONTEXTUAL_FEATURES,
        purpose="dense pool 유지 + soft rerank — 이웃 부실(ni) 처방. "
                "pool이 dense top-N에 갇히므로 nd는 개선되지 않는 방향으로 움직인다"),
    "attribute_knn": dict(
        action="attribute_knn", scope="global", tool_name="attribute_knn",
        cost=ACTION_COST["attribute_knn"], requires_features=True,
        max_features=MAX_CONTEXTUAL_FEATURES,
        purpose="hard term filter로 부분집합을 먼저 만들고 그 안에서 거리 계산 — "
                "dense 축이 못 보는 후보를 직접 복구. 라우팅 오류(cm) 처방이자 "
                "evidence novelty(neighbor_disjoint) 측정의 주 대상"),
    # ── Stage 3 이후 (scope 재도입) ──
    "contextual_knn@cluster": dict(
        action="contextual_knn", scope="cluster", tool_name="contextual_knn",
        cost=COST_CONTEXTUAL_KNN_CLUSTER, requires_features=True,
        max_features=MAX_CONTEXTUAL_FEATURES,
        purpose="cluster-scoped contextual (Stage 3)"),
}

PLANNER_STAGE_ACTIONS: Dict[str, tuple] = {
    # stage1 = 최초 오라클(0811)과 동일 — 재현/비교 기준선으로 남긴다
    "stage1": ("stop", "global_knn", "contextual_knn"),
    # stage2 = attribute_knn 추가. dense가 못 보는 축을 처음으로 관측한다
    "stage2": ("stop", "global_knn", "contextual_knn", "attribute_knn"),
    # stage3 = scope 재도입
    "stage3": ("stop", "global_knn", "contextual_knn", "attribute_knn",
               "contextual_knn@cluster"),
}


def planner_stage_actions(stage: Optional[str] = None) -> tuple:
    """현재(또는 지정) stage에서 유효한 registry key 목록."""
    return PLANNER_STAGE_ACTIONS.get((stage or PLANNER_STAGE).lower(),
                                     PLANNER_STAGE_ACTIONS["stage1"])


def planner_action_names(stage: Optional[str] = None) -> tuple:
    """모델이 출력할 수 있는 action 문자열(중복 제거) — 프롬프트/검증용."""
    return tuple(dict.fromkeys(_PLANNER_ACTION_SPECS[k]["action"]
                               for k in planner_stage_actions(stage)))


def planner_action_spec(key: str) -> Optional[Dict[str, Any]]:
    return _PLANNER_ACTION_SPECS.get(key)


def resolve_planner_action(
    action: str,
    features: Optional[List[str]] = None,
    scope: Optional[str] = None,
    candidate_pool: Optional[List[str]] = None,
    stage: Optional[str] = None,
) -> Dict[str, Any]:
    """
    모델 출력(또는 registry key)을 현재 stage에서 실행 가능한 형태로 정규화.

    Returns: {"valid", "reason", "key", "spec", "features"}

    ★ 이 함수가 "stage가 바뀌어도 서빙이 안 깨지는" 유일한 관문이다.
      예외를 던지지 않는다 — 판단은 항상 호출자(fallback 정책 보유자)가 한다.
    """
    keys = planner_stage_actions(stage)
    if not isinstance(action, str) or not action:
        return {"valid": False, "reason": "action_missing",
                "key": None, "spec": None, "features": []}

    if "@" in action:                       # registry key를 그대로 받은 경우
        action, scope = action.split("@", 1)

    if scope:
        cand = f"{action}@{scope}"
        if cand in keys:
            key = cand
        elif action in keys and _PLANNER_ACTION_SPECS[action]["scope"] == scope:
            key = action
        else:
            return {"valid": False,
                    "reason": f"scope_unsupported_in_{stage or PLANNER_STAGE}:{scope}",
                    "key": None, "spec": None, "features": []}
    else:
        key = action if action in keys else None
        if key is None:
            return {"valid": False,
                    "reason": f"action_not_in_{stage or PLANNER_STAGE}:{action}",
                    "key": None, "spec": None, "features": []}

    spec = dict(_PLANNER_ACTION_SPECS[key])
    feats: List[str] = []
    if spec["requires_features"]:
        raw = [str(f) for f in (features or []) if isinstance(f, (str, int, float))]
        allowed = set(CONTEXTUAL_CANDIDATE_FIELDS)
        if candidate_pool is not None:
            allowed &= set(candidate_pool)
        feats = [f for f in dict.fromkeys(raw) if f in allowed][:spec["max_features"]]
        if not feats:
            return {"valid": False, "reason": "no_executable_features",
                    "key": key, "spec": spec, "features": []}
    return {"valid": True, "reason": None, "key": key, "spec": spec, "features": feats}


def planner_tool_payload(
    spec: Dict[str, Any], *,
    features: Optional[List[str]] = None,
    embedding: Optional[List[float]] = None,
    vec_index: Optional[str] = None,
    transaction: Optional[Dict[str, Any]] = None,
    cluster_id: int = -1,
    top_k: int = 5,
) -> Dict[str, Any]:
    """registry spec → MCP tool 호출 payload.

    ★ contextual_knn과 attribute_knn이 **동일한 payload 계약**을 쓴다
      (features = 이름 리스트, transaction = 값 조회용 dict).
      attribute_knn이 값을 필요로 한다고 해서 호출자가 다른 형태를 만들면
      planner가 두 action을 대칭적으로 다룰 수 없다 — 값 해석은 tool 안에서 한다.
    """
    payload: Dict[str, Any] = {
        "embedding": embedding, "vec_index": vec_index, "top_k": int(top_k),
    }
    if spec["requires_features"]:
        payload["features"] = list(features or [])
        payload["transaction"] = transaction or {}
    if spec["scope"] == "cluster":
        payload["scope"] = "cluster"
        payload["cluster_id"] = int(cluster_id)
    elif spec["scope"] == "global":
        payload["scope"] = "global"
    return payload


def build_planner_action_space(
    agree_like_axes: List[str], *, stage: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    오라클 rollout이 전수 실행할 후보 목록 — stage-aware.
    stage1은 build_grpo_action_space_v1과 동일한 결과를 낸다(0811 오라클 호환).
    """
    actions: List[Dict[str, Any]] = []
    feats_sets = candidate_feature_sets(agree_like_axes)
    for key in planner_stage_actions(stage):
        spec = _PLANNER_ACTION_SPECS[key]
        if not spec["requires_features"]:
            a: Dict[str, Any] = {"action": spec["action"]}
            if spec["scope"] and key != spec["action"]:
                a["scope"] = spec["scope"]
            actions.append(a)
            continue
        for fs in feats_sets:
            a = {"action": spec["action"], "features": list(fs)}
            if key != spec["action"]:          # scope variant만 scope 명시
                a["scope"] = spec["scope"]
            actions.append(a)
    return actions


def build_planner_tool_catalog(stage: Optional[str] = None) -> str:
    """planner 프롬프트/문서용 action 카탈로그.
    (FOLLOWUP_TOOLS = legacy planner 전용, 이쪽 = 학습 planner 전용 — 분리 유지)"""
    lines = ["## PLANNER ACTIONS (select EXACTLY ONE)"]
    for key in planner_stage_actions(stage):
        spec = _PLANNER_ACTION_SPECS[key]
        sig = (f'{{"action":"{spec["action"]}","features":[...]}}'
               if spec["requires_features"] else f'{{"action":"{spec["action"]}"}}')
        lines.append(f"- {sig}   cost={spec['cost']}")
        lines.append(f"    {spec['purpose']}")
    return "\n".join(lines)


# ★ 하위호환 별칭 — controller.py / oracle_extractor.py가 이 이름을 import한다.
#   심볼을 유지한 채 stage-aware 구현으로 위임한다.
build_grpo_action_space = build_planner_action_space
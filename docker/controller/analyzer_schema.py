"""
analyzer_schema.py — Analyzer 출력 스키마 어댑터 (단일 변환 지점)
====================================================================
목적:
  local_analyzer.py(서빙 코드)의 출력 schema는 노트북(analyzer_finetune_v9
  등)의 최적화 진행 상황에 따라 계속 바뀔 수 있다. 지금 알려진 변형은
  최소 2종류:
    1) 구버전 6키: {"classification","confidence","reasoning","key_evidence",
                   "final_verdict","persona_used","tree_rules_used"}
       (local_analyzer.py 현재 코드가 실제로 반환하는 schema)
    2) 노트북 v9 3키: {"cls","sufficiency","what_missing"}
       (학습 목표 schema — sufficiency 텍스트 판단 기반 agent loop를 위해
       설계됨, 아직 local_analyzer.py에 서빙 코드로는 반영되지 않음)
  앞으로 또 바뀔 수 있다 (예: sufficiency가 confident/ambiguous가 아니라
  high/medium/low로, 또는 evidence_gap_type 같은 새 키가 추가되는 식).

설계 원칙 — "schema가 바뀌면 이 파일만 고친다":
  controller.py/orchestrator_agentic.py/agent_tools.py는 analyzer가 실제로
  어떤 키를 쓰는지 전혀 몰라야 한다. 이 모듈이 raw dict를 받아 항상 동일한
  표준 표현(NormalizedAnalysis)으로 바꿔주고, 그 외 모든 코드는
  NormalizedAnalysis의 필드(classification/is_sufficient/gap_hint/...)만
  본다. 즉 "analyzer 출력 스키마"와 "agent loop가 보는 표현"을 이 파일이
  분리한다.

스키마가 또 바뀌면 해야 할 일은 정확히 이것뿐이다:
  1) 새 schema를 식별하는 _detect_xxx(raw) -> bool 함수 작성
  2) 그 schema를 NormalizedAnalysis로 변환하는 _parse_xxx(raw) -> NormalizedAnalysis 작성
  3) _PARSERS 리스트 맨 앞에 (detect, parse) 튜플 추가 (우선순위 = 리스트 순서)
  controller.py/orchestrator_agentic.py는 단 한 줄도 고칠 필요가 없다.

"sufficiency는 텍스트로 판단" 원칙:
  gap_category 분류는 confidence 같은 숫자가 아니라 gap_hint/reasoning의
  키워드(텍스트)를 1차 신호로 사용한다. 숫자(confidence)는 텍스트 신호가
  없을 때만 보조적으로 쓴다 — 이 원칙은 스키마가 바뀌어도 _classify_gap_text
  하나에 고정되어 있으므로 흔들리지 않는다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable, Tuple
from pydantic import BaseModel, Field


# ============================================================
# 표준 내부 표현 — 이 클래스의 필드가 "agent loop가 보는 API"의 전부다.
# ============================================================

class NormalizedAnalysis(BaseModel):
    """
    analyzer의 실제 출력 schema와 무관하게, controller.py(plan 프롬프트
    생성, deterministic_plan)와 orchestrator_agentic.py(응답 조립)가
    공통으로 참조하는 표준 표현.
    """
    classification: str = "NORMAL"          # "NORMAL" | "ABNORMAL"
    is_sufficient:  bool = True              # True -> accept_verdict로 직행 가능
    gap_hint:        Optional[str] = None    # 부족 이유 원문 텍스트 (tool 매핑/사람이 읽는 용도)
    gap_category:     Optional[str] = None   # "sparse" | "mismatch" | "unclear" | None
    confidence:        Optional[int] = None  # 보조 신호. 텍스트가 우선이라 없어도 동작에 지장 없음
    reasoning_text:      str = ""            # /detect 응답의 reasoning 필드에 그대로 쓰이는 사람이 읽는 텍스트
    schema_name:           str = "unknown"   # 어느 파서가 처리했는지 (로그/메트릭/디버깅용)
    raw:                     Dict[str, Any] = Field(default_factory=dict)  # 원본 보존 (필요시 하위 호환 접근)

    # ── 구버전(6키) 코드가 .get("classification")/.get("confidence") 식으로
    #    여전히 dict 접근을 할 수 있으므로, 동일 키 이름으로도 노출해둔다.
    def to_legacy_dict(self) -> Dict[str, Any]:
        return {
            "classification": self.classification,
            "confidence": self.confidence if self.confidence is not None else (85 if self.is_sufficient else 55),
            "reasoning": self.reasoning_text,
            "key_evidence": self.raw.get("key_evidence", []),
        }


def _classify_gap_text(text: str) -> str:
    """
    gap_hint/reasoning 텍스트의 키워드로 부족 유형을 분류한다.
    숫자가 아니라 텍스트가 1차 신호라는 원칙이 고정되는 지점 — 스키마가
    바뀌어도 이 함수 하나만 보면 "어떻게 판단하는지"를 알 수 있다.
    """
    t = (text or "").lower()
    mismatch_kw = ("mismatch", "doesn't fit", "does not fit", "wrong cluster", "persona")
    sparse_kw   = ("sparse", "not enough", "cannot pin", "more neighbor", "only top-1", "top-1 seen")
    if any(k in t for k in mismatch_kw):
        return "mismatch"
    if any(k in t for k in sparse_kw):
        return "sparse"
    return "unclear"


# ============================================================
# 스키마별 detect/parse 쌍 — 새 schema 추가 시 여기에만 추가
# ============================================================

# ── 노트북 analyzer_finetune_v9 3키 schema ──
# {"cls": "NORMAL|ABNORMAL", "sufficiency": "confident|ambiguous",
#  "what_missing": "<=15 words" | null}
def _detect_3key(raw: Dict[str, Any]) -> bool:
    # "sufficiency" 키의 존재가 이 schema의 결정적 지문이다. "cls"만 있고
    # sufficiency가 없는 경우는 (아직 등록되지 않은) 다른 schema일 수
    # 있으므로 3key로 오판하지 않고 다음 파서/fallback으로 넘긴다.
    return "sufficiency" in raw


def _parse_3key(raw: Dict[str, Any]) -> NormalizedAnalysis:
    cls = str(raw.get("cls", "NORMAL")).upper()
    sufficiency = raw.get("sufficiency", "confident")
    what_missing = raw.get("what_missing")
    is_sufficient = (sufficiency != "ambiguous")
    gap_category = None if is_sufficient else _classify_gap_text(what_missing or "")
    return NormalizedAnalysis(
        classification=cls,
        is_sufficient=is_sufficient,
        gap_hint=what_missing,
        gap_category=gap_category,
        confidence=85 if is_sufficient else 55,
        reasoning_text=f"sufficiency={sufficiency}; what_missing={what_missing or '(none)'}",
        schema_name="notebook_v9_3key",
        raw=raw,
    )


# ── 구버전 6키 schema (local_analyzer.py 현재 서빙 코드) ──
# {"classification","confidence","reasoning","key_evidence",
#  "final_verdict","persona_used","tree_rules_used"}
def _detect_6key(raw: Dict[str, Any]) -> bool:
    return "classification" in raw or "confidence" in raw


def _parse_6key(raw: Dict[str, Any]) -> NormalizedAnalysis:
    cls = str(raw.get("classification", "NORMAL")).upper()
    conf = int(raw.get("confidence", 80))
    reasoning = raw.get("reasoning") or ""
    gap = _classify_gap_text(reasoning)
    # 텍스트에 sparse/mismatch 키워드가 없고 confidence도 충분히 높으면 confident.
    # (낮은 confidence만으로는 "왜" 부족한지 알 수 없어 gap_category=unclear로
    #  남기되, is_sufficient는 여전히 False로 — planner 기본값(expand)이 처리)
    is_sufficient = (gap == "unclear") and conf >= 60
    return NormalizedAnalysis(
        classification=cls,
        is_sufficient=is_sufficient,
        gap_hint=(reasoning if not is_sufficient else None),
        gap_category=(None if is_sufficient else gap),
        confidence=conf,
        reasoning_text=reasoning,
        schema_name="legacy_6key",
        raw=raw,
    )


# ============================================================
# 파서 레지스트리 — 우선순위는 리스트 순서. 새 schema는 맨 앞에 추가.
# ============================================================
_PARSERS: List[Tuple[Callable[[Dict[str, Any]], bool],
                      Callable[[Dict[str, Any]], NormalizedAnalysis]]] = [
    (_detect_3key, _parse_3key),
    (_detect_6key, _parse_6key),
]


def register_schema(
    detect: Callable[[Dict[str, Any]], bool],
    parse:  Callable[[Dict[str, Any]], NormalizedAnalysis],
    *, priority: bool = True,
) -> None:
    """
    런타임에 새 schema 파서를 등록한다(코드 수정 없이 plug-in 하고 싶을 때).
    priority=True면 맨 앞(최우선)에, False면 맨 뒤(fallback 직전)에 추가.
    일반적으로는 _PARSERS 리스트에 직접 추가하는 편이 더 명시적이라
    권장되지만, 동적 등록이 필요한 경우(예: A/B 테스트로 두 analyzer
    버전을 동시에 굴릴 때)를 위해 제공한다.
    """
    entry = (detect, parse)
    if priority:
        _PARSERS.insert(0, entry)
    else:
        _PARSERS.append(entry)


def normalize_analyzer_output(raw: Dict[str, Any]) -> NormalizedAnalysis:
    """
    analyzer 응답(raw dict, 어떤 schema든)을 받아 표준 표현으로 정규화한다.
    controller.py의 node_analyze/node_plan, orchestrator_agentic.py의
    fallback 조립부가 이 함수 하나만 거치면 된다.
    """
    if not isinstance(raw, dict) or not raw:
        return NormalizedAnalysis(
            classification="NORMAL", is_sufficient=True,
            reasoning_text="empty or invalid analyzer response; defaulting to NORMAL/confident",
            schema_name="fallback_empty", raw=raw if isinstance(raw, dict) else {},
        )
    for detect, parse in _PARSERS:
        try:
            if detect(raw):
                return parse(raw)
        except Exception:
            continue  # 이 파서가 raw 형태와 안 맞으면 다음 파서 시도
    return NormalizedAnalysis(
        classification=str(raw.get("classification") or raw.get("cls") or "NORMAL").upper(),
        is_sufficient=True,
        reasoning_text=f"unrecognized analyzer schema (keys={list(raw.keys())}); defaulting to confident",
        schema_name="fallback_unknown",
        raw=raw,
    )


def map_gap_to_tool(analysis: NormalizedAnalysis, *, mismatch_tool: str = "reroute_cross_cluster",
                     sparse_tool: str = "expand_within_cluster",
                     default_tool: str = "expand_within_cluster") -> str:
    """
    gap_category(텍스트 분류 결과) -> follow-up tool 이름 매핑.
    controller.py의 deterministic_plan과 build_prompt 양쪽이 동일 규칙을
    쓰도록 단일 함수로 고정한다 (둘이 따로 구현되어 있으면 스키마가 바뀔 때
    한쪽만 고치는 사고가 날 수 있음).
    """
    if analysis.is_sufficient:
        return "accept_verdict"
    if analysis.gap_category == "mismatch":
        return mismatch_tool
    if analysis.gap_category == "sparse":
        return sparse_tool
    return default_tool

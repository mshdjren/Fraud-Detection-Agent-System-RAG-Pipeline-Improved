"""
agent_tools.py
==============
Claude-code의 내장 tool 패턴(정의 + enabled + 실행)을 참고한
orchestrator 내부 tool 계층.

목표:
- router/retriever app 코드는 최대한 유지
- orchestrator가 "어떤 tool을 쓸지" 정책적으로 선택
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable, Awaitable, List


@dataclass
class AgentTool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    is_enabled: Callable[[], bool]
    run: Callable[..., Awaitable[Dict[str, Any]]]


def choose_tool(
    *,
    fast_track_enabled: bool,
    deep_track_enabled: bool,
    compare_track_enabled: bool,
    distance_hint: Optional[float],
    fast_threshold: float,
) -> str:
    """
    간단한 orchestrator 정책:
    1) fast track이 켜져 있고 distance_hint가 매우 크면 fast tool
    2) 그 외는 기본 deep tool
    3) compare track은 A/B 실험용으로만 별도 호출 (기본 선택 아님)
    """
    if fast_track_enabled and distance_hint is not None and distance_hint >= fast_threshold:
        return "anomaly.fast_track_distance_gate"
    if deep_track_enabled:
        return "anomaly.deep_track_two_stage"
    if compare_track_enabled:
        return "anomaly.compare_track_raw_es"
    return "anomaly.deep_track_two_stage"


def list_enabled_tools(tools: List[AgentTool]) -> List[str]:
    return [t.name for t in tools if t.is_enabled()]


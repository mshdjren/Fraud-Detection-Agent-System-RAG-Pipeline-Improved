"""
judge.py — LLM-as-Judge (공용: NB03 심화평가 + evaluation harness)
==================================================================
설계 원칙:
  0. judge의 채점 대상은 what_missing 하나 — cls는 AUROC(deterministic),
     sufficiency는 nd-band 일치(deterministic)로 평가하므로 judge 불필요.
  1. judge ≠ student, judge > student — teacher(14B) 또는 Gemini만 허용,
     self-judge(0.5B/3B로 채점) 금지.
  2. 이산 rubric — 0 / 0.5 / 1 (연속 0~1보다 일관성↑).
  3. 신뢰성 검증 3종을 모듈에 내장:
       (a) judge_gt_correlation  — GT 케이스로 judge 점수 ↔ 정답 상관 (핵심)
       (b) consistency_check    — 동일 입력 2회 채점 일치율 (temperature 0)
       (c) position_bias_check  — A/B 순서 교환 시 판정 유지율

의존성 최소화: gemini는 google-generativeai, hf는 caller가 pipeline 주입.
NB(Colab)와 harness(로컬/클러스터) 양쪽에서 import 가능해야 한다.
rubric 텍스트는 이 모듈이 단일 소스 — NB에 복사하지 말고 import할 것.
"""
from __future__ import annotations
import json
import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ============================================================
# Rubric (이산 0 / 0.5 / 1) — 버전 관리 대상
# ============================================================
RUBRIC_VERSION = "v1"

WM_RUBRIC = """You are grading a fraud-detection model's `what_missing` output.
`what_missing` should specify WHAT additional retrieval evidence is needed,
grounded in the actual transaction features shown in the input.

Score with EXACTLY one of: 0, 0.5, 1
  1   = names a SPECIFIC feature or evidence gap that is actually deviant/
        ambiguous in the input (e.g. "purchase_value is 2.3σ above cluster
        mean; need more same-cluster purchase evidence").
  0.5 = plausible but generic — mentions a real category of gap without
        tying it to a concrete input feature.
  0   = boilerplate ("boundary distance; expand evidence"), restates the
        distance value, or references features NOT present in the input.

Respond ONLY with JSON: {"score": <0|0.5|1>, "reason": "<one sentence>"}"""

_VALID_SCORES = (0.0, 0.5, 1.0)


def _parse_score_json(raw: str) -> Dict:
    """judge 응답에서 JSON 파싱 + score를 이산값으로 강제."""
    t = raw.strip()
    if "```" in t:
        t = re.sub(r"```(?:json)?", "", t).strip()
    if "{" in t:
        t = "{" + t.split("{", 1)[1]
    if "}" in t:
        t = t.rsplit("}", 1)[0] + "}"
    d = json.loads(t)
    s = float(d.get("score", -1))
    # 가장 가까운 이산값으로 snap (judge가 0.7 등 내보내는 경우 방어)
    d["score"] = min(_VALID_SCORES, key=lambda v: abs(v - s)) if 0 <= s <= 1 else -1.0
    return d


# ============================================================
# Judge 백엔드 — call_fn: (prompt:str) -> str (raw text)
# ============================================================

def make_gemini_caller(api_key: str, model: str = "gemini-2.5-flash-lite") -> Callable[[str], str]:
    """Gemini judge caller. temperature=0 고정 (일관성 체크 전제)."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    mdl = genai.GenerativeModel(
        model_name=model,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0, max_output_tokens=128,
            response_mime_type="application/json",
        ),
    )
    def _call(prompt: str) -> str:
        return mdl.generate_content(prompt).text
    return _call


def make_hf_caller(hf_pipeline) -> Callable[[str], str]:
    """HF teacher(예: Qwen2.5-14B) judge caller — 기존 hf_pipeline 재사용.
    do_sample=False → deterministic (temperature 0 상당)."""
    def _call(prompt: str) -> str:
        out = hf_pipeline(
            [{"role": "user", "content": prompt + "\nIMPORTANT: Respond ONLY with valid JSON."}],
            max_new_tokens=128, do_sample=False,
        )
        return out[0]["generated_text"][-1]["content"]
    return _call


# ============================================================
# 채점 함수
# ============================================================

def judge_what_missing(call_fn: Callable[[str], str],
                       user_prompt: str, what_missing: Optional[str]) -> Dict:
    """what_missing 1건 채점. 빈 값은 judge 호출 없이 0점 (boilerplate 이하)."""
    if not what_missing or not str(what_missing).strip():
        return {"score": 0.0, "reason": "empty what_missing", "judged": False}
    prompt = (f"{WM_RUBRIC}\n\n## Input (truncated):\n{user_prompt[:800]}\n\n"
              f"## what_missing:\n{what_missing}")
    try:
        d = _parse_score_json(call_fn(prompt))
        d["judged"] = True
        return d
    except Exception as e:
        return {"score": -1.0, "reason": f"judge_error: {e}", "judged": False}


# ============================================================
# 신뢰성 검증 3종 (PASTED 문서 1의 핵심)
# ============================================================

def judge_gt_correlation(scores: List[float], gt_correct: List[int]) -> Dict:
    """
    (a) GT 교차검증 — 가장 중요.
    scores:     judge 점수 (0/0.5/1), 채점 실패(-1) 제외 후 전달
    gt_correct: 같은 케이스에서 모델 판정이 GT와 일치했는지 (0/1)

    point-biserial 상관 (= Pearson, 한쪽이 이진). 상관이 낮으면
    "judge가 좋다고 한 출력이 실제로는 안 맞는다" → rubric 수정 신호.
    """
    s = np.asarray(scores, dtype=float)
    g = np.asarray(gt_correct, dtype=float)
    mask = s >= 0
    s, g = s[mask], g[mask]
    if len(s) < 5 or s.std() == 0 or g.std() == 0:
        return {"corr": None, "n": int(len(s)),
                "note": "insufficient/degenerate data for correlation"}
    corr = float(np.corrcoef(s, g)[0, 1])
    return {
        "corr": round(corr, 4), "n": int(len(s)),
        "mean_score_when_correct":   round(float(s[g == 1].mean()), 3) if (g == 1).any() else None,
        "mean_score_when_incorrect": round(float(s[g == 0].mean()), 3) if (g == 0).any() else None,
        # 산점도용 raw (harness_report가 사용)
        "_scores": s.tolist(), "_gt_correct": g.astype(int).tolist(),
    }


def consistency_check(call_fn: Callable[[str], str],
                      cases: List[Tuple[str, str]], judge_fn=judge_what_missing) -> Dict:
    """(b) 동일 입력 2회 채점 일치율. cases: [(user_prompt, what_missing), ...]"""
    agree = total = 0
    for up, wm in cases:
        s1 = judge_fn(call_fn, up, wm).get("score", -1)
        s2 = judge_fn(call_fn, up, wm).get("score", -1)
        if s1 < 0 or s2 < 0:
            continue
        total += 1
        agree += int(s1 == s2)
    rate = agree / max(total, 1)
    return {"agreement_rate": round(rate, 4), "n": total}


def position_bias_check(call_fn: Callable[[str], str],
                        pairs: List[Tuple[str, str, str]]) -> Dict:
    """
    (c) A/B 비교 순서 편향. pairs: [(user_prompt, wm_A, wm_B), ...]
    A-then-B와 B-then-A 두 순서로 "which is better?"를 묻고 판정 유지율 측정.
    """
    tmpl = """Compare two `what_missing` outputs for the same input. Which is more
grounded in specific input features? Respond ONLY with JSON: {{"winner": "A"|"B"|"tie"}}

## Input (truncated):
{up}

## Output A:
{a}

## Output B:
{b}"""
    consistent = total = 0
    for up, wa, wb in pairs:
        try:
            r1 = json.loads(re.sub(r"```(?:json)?", "", call_fn(
                tmpl.format(up=up[:600], a=wa, b=wb))).strip())
            r2 = json.loads(re.sub(r"```(?:json)?", "", call_fn(
                tmpl.format(up=up[:600], a=wb, b=wa))).strip())
        except Exception:
            continue
        total += 1
        w1, w2 = r1.get("winner"), r2.get("winner")
        # 순서 교환 시 승자 라벨도 뒤집혀야 일관
        flipped = {"A": "B", "B": "A", "tie": "tie"}.get(w2, "?")
        consistent += int(w1 == flipped)
    return {"position_consistency": round(consistent / max(total, 1), 4), "n": total}

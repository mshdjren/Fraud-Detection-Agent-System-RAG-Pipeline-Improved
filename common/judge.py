"""
judge.py v2 — LLM-as-Judge (프로그램 지표로 검증 불가능한 2축만)
==================================================================
설계 원칙 (v1에서 변경):
  0. 채점 대상은 what_missing의 **rationale**. type/axes/hallucination/channel은
     전부 결정론 GT로 검증되므로 judge 대상에서 제외한다.
  1. judge의 존재 이유는 두 가지뿐:
       (P1) reward의 `justify` 정규식 프록시가 실제 의미를 재는지 (construct validity)
       (P2) 정규식이 못 보는 의미 오류/무관 진술 탐지 (reward hacking 감사)
  2. judge ≠ student, judge > student — teacher(14B) 또는 Gemini만.
  3. 이산 rubric 0 / 0.5 / 1.
  4. GT 상관은 cls(gt_label)가 아니라 **프로그램 proxy(justify)와의 일치율**로 잰다.
"""
from __future__ import annotations
import json, re
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

RUBRIC_VERSION = "v2"
_VALID_SCORES = (0.0, 0.5, 1.0)

# ── P1: licensing 조건 진술 완전성 (justify의 semantic ground truth) ──
WM_CONDITION_RUBRIC = """You grade the RATIONALE (text after "—") of a retrieval-quality
verdict. Each verdict type is licensed by specific conditions shown in the input.
Grade ONLY whether the rationale states the conditions that license the stated type.

  neighbor_insufficient — licensed by BOTH (a) neighbors are spread out AND
                          (b) at least 2 features differ from neighbors.
      1   = states BOTH (a) and (b)
      0.5 = states exactly one of them
      0   = states neither, or asserts neighbors are dense/consistent (contradiction)
  cluster_mismatch      — licensed by a deviant numeric z-score OR a rare categorical value.
      1   = names the deviation with its magnitude/value
      0.5 = mentions deviation generically without magnitude or value
      0   = no deviation stated
  accept                — licensed by BOTH (a) features in range / matching the persona
                          AND (b) neighbors dense OR fewer than 2 features differing.
      1   = states BOTH   0.5 = one   0   = neither, or contradicts the input

Respond ONLY with JSON: {"score": <0|0.5|1>, "stated": ["<condition>", ...], "reason": "<one sentence>"}"""

# ── P2: 인용 축 ↔ rationale 인과 정합 (정규식이 못 보는 축) ──
WM_COHERENCE_RUBRIC = """You audit whether a rationale is CAUSALLY CONNECTED to the
features it cites, using only the input shown. Ignore style, length and wording.

  1   = the rationale explains why THOSE cited features make the verdict hold,
        and every claim it makes is true of the input
  0.5 = generic but not false — could be attached to any sample with the same verdict
  0   = makes a claim contradicted by the input, or the reasoning is about features
        other than the ones cited

Respond ONLY with JSON: {"score": <0|0.5|1>, "generic": <true|false>, "reason": "<one sentence>"}"""


def _parse_score_json(raw: str) -> Dict:
    t = raw.strip()
    if "```" in t: t = re.sub(r"```(?:json)?", "", t).strip()
    if "{" in t: t = "{" + t.split("{", 1)[1]
    if "}" in t: t = t.rsplit("}", 1)[0] + "}"
    d = json.loads(t)
    s = float(d.get("score", -1))
    d["score"] = min(_VALID_SCORES, key=lambda v: abs(v - s)) if 0 <= s <= 1 else -1.0
    return d


# ── 백엔드 (v1과 동일) ──
def make_gemini_caller(api_key: str, model: str = "gemini-2.5-flash-lite") -> Callable[[str], str]:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    mdl = genai.GenerativeModel(model_name=model,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0, max_output_tokens=192, response_mime_type="application/json"))
    return lambda prompt: mdl.generate_content(prompt).text


def make_hf_caller(hf_pipeline):
    def _call(prompt: str) -> str:
        out = hf_pipeline([{"role": "user", "content": prompt + "\nIMPORTANT: Respond ONLY with valid JSON."}])
        return out[0]["generated_text"][-1]["content"]     # 생성 인자는 generation_config가 담당
    return _call

# ── 채점 ──
def judge_rationale(call_fn, user_prompt: str, wm_full: str, axis: str = "condition") -> Dict:
    """wm_full = '<type>: <evidence> — <rationale>' 원문 그대로."""
    if not wm_full or not str(wm_full).strip():
        return {"score": 0.0, "reason": "empty", "judged": False}
    rubric = WM_CONDITION_RUBRIC if axis == "condition" else WM_COHERENCE_RUBRIC
    prompt = (f"{rubric}\n\n## Input (truncated):\n{user_prompt[:1200]}\n\n"
              f"## Model verdict:\n{wm_full}")
    try:
        d = _parse_score_json(call_fn(prompt)); d["judged"] = True; d["axis"] = axis
        return d
    except Exception as e:
        return {"score": -1.0, "reason": f"judge_error: {e}", "judged": False, "axis": axis}


# ── (a) 핵심: reward proxy 타당성 — cls GT가 아니라 justify와 대조 ──
def judge_proxy_agreement(judge_scores: List[float], proxy_scores: List[float],
                          tol: float = 0.25) -> Dict:
    """judge(의미) vs justify(정규식). 상관이 낮으면 reward가 다른 걸 재는 중."""
    j = np.asarray(judge_scores, float); p = np.asarray(proxy_scores, float)
    m = j >= 0; j, p = j[m], p[m]
    if len(j) < 5 or j.std() == 0 or p.std() == 0:
        return {"corr": None, "n": int(len(j)), "note": "degenerate"}
    return {"corr": round(float(np.corrcoef(j, p)[0, 1]), 4),
            "agreement": round(float((np.abs(j - p) <= tol).mean()), 4),
            "n": int(len(j)),
            "judge_high_proxy_low": int(((j >= 0.75) & (p <= 0.25)).sum()),   # 정규식 miss
            "judge_low_proxy_high": int(((j <= 0.25) & (p >= 0.75)).sum()),   # ★ hacking 후보
            "_judge": j.tolist(), "_proxy": p.tolist()}


# ── (b) 일관성 / (c) 순서편향 (v1 로직 유지, 시그니처만 정리) ──
def consistency_check(call_fn, cases: List[Tuple[str, str]], axis: str = "condition") -> Dict:
    agree = total = 0
    for up, wm in cases:
        s1 = judge_rationale(call_fn, up, wm, axis).get("score", -1)
        s2 = judge_rationale(call_fn, up, wm, axis).get("score", -1)
        if s1 < 0 or s2 < 0: continue
        total += 1; agree += int(s1 == s2)
    return {"agreement_rate": round(agree / max(total, 1), 4), "n": total}


def position_bias_check(call_fn, pairs: List[Tuple[str, str, str]]) -> Dict:
    tmpl = """Compare two retrieval-quality verdicts for the same input. Which rationale
better states the conditions that license its verdict? Respond ONLY with JSON:
{{"winner": "A"|"B"|"tie"}}

## Input (truncated):
{up}

## Output A:
{a}

## Output B:
{b}"""
    consistent = total = 0
    for up, wa, wb in pairs:
        try:
            r1 = json.loads(re.sub(r"```(?:json)?", "", call_fn(tmpl.format(up=up[:800], a=wa, b=wb))).strip())
            r2 = json.loads(re.sub(r"```(?:json)?", "", call_fn(tmpl.format(up=up[:800], a=wb, b=wa))).strip())
        except Exception:
            continue
        total += 1
        flipped = {"A": "B", "B": "A", "tie": "tie"}.get(r2.get("winner"), "?")
        consistent += int(r1.get("winner") == flipped)
    return {"position_consistency": round(consistent / max(total, 1), 4), "n": total}
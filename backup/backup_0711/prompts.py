"""
prompts.py — 두 노트북 공통 프롬프트/포맷터 레이어.

원본 Section 5(cell 19/23/24/26)를 외부화. agent 버전과 forwardRAG 버전의
prompt builder를 모두 보유하므로, 두 노트북이 같은 모듈을 공유한다.

런타임 데이터(coreset 맵, distance 백분위수, test_df 등)는 data_io.load_all()이
계산하여 bind_runtime()으로 이 모듈에 주입한다 — 그 덕분에 아래 함수 본문은
원본과 동일하게 bare-name 전역을 참조해도 동작한다(본문 수정 0).

사용:
    import prompts
    prompts.set_mode("agent")        # 또는 "forward"
    # (data_io.load_all() 내부에서 prompts.bind_runtime(...) 호출됨)
    sys_p, usr_p = prompts.build_sample(idx, for_teacher=True, include_gt=True)
"""
import numpy as np
from typing import Dict, List, Optional, Tuple

NUMERIC_COLS     = ["purchase_value", "age"]
CATEGORICAL_COLS = ["sex", "source", "browser", "IP_country",
                    "month_purchase", "weekday_purchase"]
FEATURE_COLS     = NUMERIC_COLS + CATEGORICAL_COLS
# ============================================================
# 런타임 주입 전역 (data_io.load_all → bind_runtime 가 채움)
# ============================================================
DIST_P10 = DIST_P25 = DIST_P50 = DIST_P75 = DIST_P90 = DIST_IQR = 0.0
NORM_CLOSE_THRESHOLD = NORM_MODERATE_THRESHOLD = 0.0
coreset_feature_map: Dict[int, Dict] = {}
coreset_emb_map: Dict[int, np.ndarray] = {}
CLUSTER_PERSONAS: Dict[int, Dict] = {}
test_df = None
test_emb = None
labels = None
gt_clusters = None
top5_indices = None


def bind_runtime(**kw):
    """data_io 가 계산한 런타임 상태를 이 모듈 전역으로 주입."""
    globals().update(kw)


# ============================================================
# SYSTEM PROMPT (양 노트북 동일)
# ============================================================
SYSTEM_PROMPT = (
    "You are an expert fraud detection analyst. "
    "Analyze the TEST TRANSACTION by comparing it with the cluster profile "
    "and retrieved similar normal patterns. "
    "\n\n"
    "CRITICAL: Respond in ENGLISH ONLY with valid JSON. "
    "Do not use Chinese or any other language."
)


# ============================================================
# 포맷터 (cell 19) — 순수 함수
# ============================================================
def format_persona(persona: Dict) -> str:
    if not persona:
        return ""
    lines = []
    if "description" in persona:
        lines.append(f"**Summary**: {persona['description']}")
    if "numeric_stats" in persona:
        lines.append("\n**Numeric Statistics:**")
        for feat, stats in persona["numeric_stats"].items():
            lines.append(
                f"  - {feat}: mean={stats.get('mean',0):.3f}, "
                f"std={stats.get('std',0):.3f}, "
                f"min={stats.get('min',0):.3f}, max={stats.get('max',0):.3f}"
            )
    if "categorical_dist" in persona:
        lines.append("\n**Categorical Distribution:**")
        for feat, dist in persona["categorical_dist"].items():
            dominant = max(dist, key=dist.get) if dist else "?"
            pct = int(dist.get(dominant, 0) / max(sum(dist.values()), 1) * 100)
            details = ", ".join(f"{k}:{v}" for k, v in list(dist.items())[:4])
            lines.append(f"  - {feat}: dominant={dominant} ({pct}%)  [{details}]")
    return "\n".join(lines)


# ── z-score 언어화 (LLM의 산술비교 병목 우회) ─────────────────
# raw 통계(mean/std)를 주고 '비교하라' 대신, 편차를 미리 계산해 문장으로 준다.
def verbalize_z(val, mean, std):
    if val is None or not std:
        return None
    z = (val - mean) / std
    a = abs(z); d = "above" if z >= 0 else "below"
    if a < 1.0:   tag = f"within range ({z:+.1f}σ)"
    elif a < 2.0: tag = f"mildly off ({z:+.1f}σ {d} mean)"
    else:         tag = f"EXTREME ({z:+.1f}σ {d} mean)"
    return tag


def format_persona_zscore(test_features: Dict, persona: Dict,
                          numeric_only=("purchase_value", "age")) -> str:
    """persona 통계 + test 값을 z-score 문장으로 변환. 수치형 2개만(범주형 무신호 제거)."""
    if not persona or "numeric_stats" not in persona:
        return "(no cluster profile)"
    nst = persona["numeric_stats"]
    head = persona.get("description", "")
    lines = [f"**Cluster**: {head}" if head else "**Cluster profile (z vs cluster):**"]
    for f in numeric_only:
        s = nst.get(f)
        if not s:
            continue
        tag = verbalize_z(test_features.get(f), s.get("mean", 0), s.get("std", 0))
        if tag:
            lines.append(f"  - {f}: {tag}")
    return "\n".join(lines)


def format_transaction(data: Dict) -> str:
    sections = {
        "Numeric Features":     ["purchase_value", "age"],
        "Categorical Features": ["sex", "source", "browser"],
        "Additional Context":   ["weekday_purchase", "month_purchase", "IP_country"],
    }
    out = []
    for sec_name, cols in sections.items():
        vals = [(c, data.get(c)) for c in cols if c in data and data.get(c) is not None]
        if not vals:
            continue
        out.append(f"[{sec_name}]")
        for col, val in vals:
            fmt = f"{round(val, 1)}" if isinstance(val, float) else str(val)
            out.append(f"  - {col}: {fmt}")
        out.append("")
    return "\n".join(out).strip()


# ============================================================
# Distance 함수 (cell 23) — 런타임 전역 참조
# ============================================================
def normalize_dist(d: float) -> float:
    if DIST_IQR == 0:
        return 0.5
    raw = (d - DIST_P50) / DIST_IQR
    return float(1 / (1 + np.exp(-raw)))


def dist_tag(d: float) -> str:
    if d < NORM_CLOSE_THRESHOLD:
        return "close match ✓"
    elif d < NORM_MODERATE_THRESHOLD:
        return "moderate match"
    else:
        return "distant match ⚠"


def compute_l2_distances(test_emb_vec: np.ndarray,
                         top5_orig_indices: List[int]) -> List[float]:
    distances = []
    for oi in top5_orig_indices:
        ref_emb = coreset_emb_map.get(int(oi))
        if ref_emb is not None:
            d = float(np.linalg.norm(
                test_emb_vec.astype(np.float32) - ref_emb.astype(np.float32)
            ))
        else:
            d = float("inf")
        distances.append(d)
    return distances


def format_top5_neighbors(top5_orig_indices: List[int],
                          distances: List[float]) -> str:
    if not top5_orig_indices:
        return "No retrieved results."
    top1_raw  = distances[0] if distances else float("inf")
    top1_norm = normalize_dist(top1_raw)
    tag       = dist_tag(top1_raw)
    # ── 이웃 spread: boundary에서 distance와 독립인 정방향 구조신호 ──
    finite = [d for d in distances if d != float("inf")]
    if len(finite) > 1:
        cv = float(np.std(finite)) / float(np.mean(finite)) if np.mean(finite) > 1e-9 else 0.0
        sp_tag = "neighbors spread out ⚠" if cv > 0.15 else "neighbors tightly agree ✓"
        spread_line = f"Neighbor spread: {sp_tag}"
    else:
        spread_line = "Neighbor spread: n/a (only top-1)"
    # cls 판단축은 normalized distance 1개면 충분 (raw L2 4자리·분포참조 제거)
    lines = [f"Top-1 distance(norm): {top1_norm:.2f}  ({tag})", spread_line, ""]
    # 이웃은 feature 비교용 — index/거리 중복 제거, feature 값만 1자리 반올림
    for i, oi in enumerate(top5_orig_indices, 1):
        feat = coreset_feature_map.get(int(oi), {})
        feat_str = ", ".join(
            f"{k}={round(v, 1) if isinstance(v, float) else v}"
            for k, v in feat.items() if v is not None
        )
        if feat_str:
            lines.append(f"Neighbor {i}: {feat_str}")
    return "\n".join(lines)


# ============================================================
# extract_test_features (cell 26) — test_df 런타임 참조
# ============================================================
def extract_test_features(idx: int) -> Dict:
    row = test_df.iloc[idx]
    return {col: (None if (row[col] != row[col]) else row[col])
            for col in FEATURE_COLS if col in test_df.columns}


# ============================================================
# AGENT 버전 prompt builders (v9 cell 24)
# ============================================================
def build_analyzer_user_prompt_agent(
    test_features:     Dict,
    cluster_id:        int,
    top5_orig_indices: List[int],
    distances:         List[float],
    persona:           Optional[Dict] = None,
    top_k:             int = 1,
) -> str:
    """Inference용 — GT 미포함. top_k로 evidence 개수 가변 (agent loop 지원)."""
    top_k = max(1, min(top_k, 3))
    k_oi   = top5_orig_indices[:top_k]
    k_dist = distances[:top_k]

    test_fmt    = format_transaction(test_features)
    top_fmt     = format_top5_neighbors(k_oi, k_dist)
    persona_fmt = format_persona(persona) if persona else None

    sections = [f"## TEST TRANSACTION\n{test_fmt}"]
    cluster_block = [f"## CLUSTER #{cluster_id}"]
    # raw 통계 덤프 대신 z-score 언어화(수치형 2개) — 산술비교 병목 우회
    cluster_block.append(format_persona_zscore(test_features, persona)
                         if persona else f"Cluster ID: {cluster_id} (no profile)")
    sections.append("\n".join(cluster_block))
    sections.append(f"## RETRIEVED NORMAL NEIGHBORS (top-{top_k})\n{top_fmt}")
    sections.append("""You decide the verdict AND explain what is uncertain.

- cls: use the normalized distance (nd) as the PRIMARY basis — low nd → NORMAL,
  high nd → ABNORMAL. Restating nd for cls is fine; nd is the right signal for it.
- sufficiency: "ambiguous" if nd is mid-range OR the neighbors are spread out;
  otherwise "confident".
- next_action (only if ambiguous): "expand" if neighbors are spread / too few;
  "reroute" if the sample does not fit this cluster's persona; else "none".
- what_missing (the part only you can do): when ambiguous, explain in PLAIN
  FEATURE / PERSONA language WHICH concrete attributes make this case uncertain
  (e.g. "purchase_value far above cluster range and IP_country mismatch"), so a
  re-search can target them. Name features, not the distance number. Null if confident.

Output ONLY compact JSON (no markdown):
{"cls":"NORMAL|ABNORMAL","sufficiency":"confident|ambiguous","next_action":"none|expand|reroute","what_missing":"<=15 words naming concrete features, or null","reason":"<=20 words"}""")
    return "\n\n".join(sections)


def build_teacher_prompt_gtaware_agent(
    test_features, cluster_id, top5_orig_indices, distances,
    gt_label, persona=None, top_k=1,
):
    """[실험] GT-aware what_missing 생성판.
    GT를 cls 결정이 아니라 'why this label'의 feature 설명 방향에 사용.
    → GT-free보다 일관된 feature 명세를 유도(2번 답의 방안)."""
    base = build_analyzer_user_prompt_agent(
        test_features, cluster_id, top5_orig_indices, distances, persona, top_k=top_k)
    gt = "ABNORMAL" if gt_label == 1 else "NORMAL"
    if gt_label == 1:
        ask = ("name the 2-3 features that make it ABNORMAL — the attributes whose "
               "TEST values are FARTHEST from the cluster mean/range or mismatch the "
               "neighbor (e.g. 'purchase_value far below range, IP_country mismatch').")
    else:
        ask = ("name the 2-3 features that keep it NORMAL — the attributes whose TEST "
               "values sit CLOSEST to the cluster mean/range (e.g. 'purchase_value and "
               "age within range, source matches').")
    hint = (
        f"\n\nThis transaction is {gt}.\n"
        f"`cls`={gt} (copy). Your task is `what_missing`: {ask}\n"
        f"- Compare TEST feature values to the cluster stats and the neighbor's "
        f"features. Pick the features that ACTUALLY deviate (for ABNORMAL) or match "
        f"(for NORMAL) — do not list features that are unremarkable.\n"
        f"- ALWAYS fill what_missing with concrete feature names; never the distance "
        f"number, never 'boundary distance'.\n"
        f"- `reason` (<=20 words): one concrete feature comparison.\n"
        f"- `sufficiency`/`next_action`: ignore — overwritten by a rule."
    )
    return base + hint


def build_teacher_prompt_with_gt_agent(
    test_features:     Dict,
    cluster_id:        int,
    top5_orig_indices: List[int],
    distances:         List[float],
    gt_label:          int,
    persona:           Optional[Dict] = None,
    include_gt_hint:   bool = False,
    top_k:             int = 1,
) -> str:
    """Teacher 호출용(agent) — GT hint 선택 가능."""
    base = build_analyzer_user_prompt_agent(
        test_features, cluster_id, top5_orig_indices, distances, persona, top_k=top_k
    )
    if not include_gt_hint:
        return base

    gt_str = "ABNORMAL" if gt_label == 1 else "NORMAL"
    gt_hint = (
        f"\n\nThe verdict is ALREADY decided: this transaction is {gt_str}. "
        f"Do not deliberate about cls — just copy it.\n"
        f"Your ONLY real task is `what_missing`: name the 2-3 CONCRETE FEATURES "
        f"whose values look off vs the cluster (compare the TEST feature values to "
        f"the cluster mean/range and to the neighbor's features). ALWAYS fill it "
        f"with feature names (e.g. 'purchase_value below range, IP_country "
        f"mismatch') — never leave it empty, never write 'boundary distance', never "
        f"mention the distance number.\n"
        f"- `cls` = {gt_str}.\n"
        f"- `reason` (<=20 words): cite ONE concrete feature comparison.\n"
        f"- `sufficiency`/`next_action`: ignore — they are overwritten by a rule.\n"
        f"- State only TRUE feature relations; do not invent them to fit {gt_str}."
    )
    return base + gt_hint


# ============================================================
# forwardRAG 버전 prompt builders (v6 cell 24)
# ============================================================
def build_analyzer_user_prompt_forward(
    test_features:     Dict,
    cluster_id:        int,
    top5_orig_indices: List[int],
    distances:         List[float],
    persona:           Optional[Dict] = None,
) -> str:
    """Inference용(forwardRAG) — GT label 미포함 순수 입력 프롬프트."""
    test_fmt    = format_transaction(test_features)
    top5_fmt    = format_top5_neighbors(top5_orig_indices, distances)
    persona_fmt = format_persona(persona) if persona else None

    sections = [f"## TEST TRANSACTION\n{test_fmt}"]
    cluster_block = [f"## CLUSTER #{cluster_id}"]
    if persona_fmt:
        cluster_block.extend([persona_fmt])
    else:
        cluster_block.append(f"Cluster ID: {cluster_id} (no profile)")
    sections.append("\n".join(cluster_block))
    sections.append(f"## SIMILAR NORMAL PATTERNS\n{top5_fmt}")
    sections.append("""Classify this transaction as NORMAL or ABNORMAL.

Analysis framework:
1. Statistical Deviation: Compare test values against cluster numeric/categorical profile
2. Pattern Similarity: Evaluate normalized L2 distance
   - norm_dist < 0.3: Very similar to normal patterns (likely NORMAL)
   - 0.3 ≤ norm_dist < 0.5: Moderate similarity (examine features carefully)
   - norm_dist ≥ 0.5: Weak similarity (potentially ABNORMAL)

Decision criteria:
- ABNORMAL if: significant deviation from cluster AND high normalized distance
- NORMAL if: aligns with cluster patterns AND low normalized distance
- Consider both statistical deviation and distance together
- Confidence: 0=uncertain, 100=certain
- Reference specific feature values and normalized distance in reasoning

Respond with ONLY this JSON format (no markdown, no extra text):
{
  "classification": "NORMAL or ABNORMAL",
  "confidence": 0-100,
  "reasoning": "Brief analysis following framework above",
  "key_evidence": ["Point 1", "Point 2", "Point 3"],
  "final_verdict": "One concise sentence summary"
}""")
    return "\n\n".join(sections)


def build_teacher_prompt_with_gt_forward(
    test_features:     Dict,
    cluster_id:        int,
    top5_orig_indices: List[int],
    distances:         List[float],
    gt_label:          int,
    persona:           Optional[Dict] = None,
    include_gt_hint:   bool = False,
) -> str:
    """Teacher 호출용(forwardRAG) — GT hint 선택 가능."""
    base = build_analyzer_user_prompt_forward(
        test_features, cluster_id, top5_orig_indices, distances, persona
    )
    if not include_gt_hint:
        return base

    gt_str = "ABNORMAL" if gt_label == 1 else "NORMAL"
    gt_hint = (
        f"\n\nGROUND TRUTH: {gt_str}\n\n"
        f"Generate reasoning explaining WHY this is {gt_str}.\n\n"
        f"Confidence calibration:\n"
        f"- Strong evidence (cluster + distance align) → 80-95%\n"
        f"- Moderate evidence (some contradictions) → 60-80%\n"
        f"- Weak evidence (conflicting signals) → 50-70%\n"
        f"- Use normalized distance as key signal\n\n"
        f"Express uncertainty when context conflicts with label. "
        f"Cite specific distance and feature values."
    )
    return base + gt_hint


# ============================================================
# 모드 스위치 + build_sample (양 노트북 공유)
# ============================================================
_MODE = "agent"
_analyzer_builder = build_analyzer_user_prompt_agent
_teacher_builder  = build_teacher_prompt_with_gt_agent


def set_mode(mode: str):
    """노트북이 어느 prompt 변형을 쓸지 선택. "agent" | "forward"."""
    global _MODE, _analyzer_builder, _teacher_builder
    assert mode in ("agent", "forward")
    _MODE = mode
    if mode == "agent":
        _analyzer_builder = build_analyzer_user_prompt_agent
        _teacher_builder  = build_teacher_prompt_with_gt_agent
    else:
        _analyzer_builder = build_analyzer_user_prompt_forward
        _teacher_builder  = build_teacher_prompt_with_gt_forward


def build_analyzer_user_prompt(*a, **kw):
    """활성 모드의 analyzer prompt builder로 위임 (원본 함수명 호환)."""
    return _analyzer_builder(*a, **kw)


def build_teacher_prompt_with_gt(*a, **kw):
    """활성 모드의 teacher prompt builder로 위임 (원본 함수명 호환)."""
    return _teacher_builder(*a, **kw)


def build_sample(test_idx: int, for_teacher: bool = True, include_gt: bool = False):
    gt_label      = int(labels[test_idx])
    cluster_id    = int(gt_clusters[test_idx])
    top5_oi       = [int(x) for x in top5_indices[test_idx]]
    emb_vec       = test_emb[test_idx]
    persona       = CLUSTER_PERSONAS.get(cluster_id)
    test_features = extract_test_features(test_idx)
    distances     = compute_l2_distances(emb_vec, top5_oi)

    if for_teacher:
        if _MODE == "agent":
            prompt = _teacher_builder(test_features, cluster_id, top5_oi, distances,
                                      gt_label, persona, include_gt_hint=include_gt)
        else:
            prompt = _teacher_builder(test_features, cluster_id, top5_oi, distances,
                                      gt_label, persona, include_gt_hint=include_gt)
    else:
        prompt = _analyzer_builder(test_features, cluster_id, top5_oi, distances, persona)
    return SYSTEM_PROMPT, prompt
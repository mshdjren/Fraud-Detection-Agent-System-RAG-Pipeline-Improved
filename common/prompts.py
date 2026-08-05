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
import os as _os

PROMPT_VERSION = _os.getenv("PROMPT_VERSION", "v1").lower()
AGENT_OUTPUT_KEYS = _os.getenv("AGENT_OUTPUT_KEYS", "1key").lower()
# ★ [v4-compact] 동적 블록만 압축. instruction/SYSTEM 은 KV 캐시 영역이라
#   줄여도 prefill 이득이 0 이므로 손대지 않는다(정확도 리스크만 발생).
#   실측: prompt 589 = kv_reused 345(무료) + prefill 244(22ms/token).
# ★ [v4] 압축 레벨. off | safe | mid | aggressive
#   off        : v3 원문
#   safe       : nd(top1) 제거 + TEST 섹션헤더 제거   (ni 증거 전부 보존)
#   mid        : safe + CLUSTER FIT 의 value= 제거 + PERSONA 'ok' 행 필터
#   aggressive : mid + PERSONA dom 값 제거 + spread 수치 제거  ← acc 회귀 발생
PROMPT_COMPACT = _os.getenv("PROMPT_COMPACT", "off").strip().lower()
if PROMPT_COMPACT in ("true", "1", "yes"):
    PROMPT_COMPACT = "aggressive"        # 하위호환
elif PROMPT_COMPACT in ("false", "0", "no"):
    PROMPT_COMPACT = "off"
_COMPACT_ON = PROMPT_COMPACT != "off"

NUMERIC_COLS     = ["purchase_value", "age"]
CATEGORICAL_COLS = ["sex", "source", "browser", "IP_country",
                    "month_purchase", "weekday_purchase"]
FEATURE_COLS     = NUMERIC_COLS + CATEGORICAL_COLS

Z_TAU = 1.5
NI_SPREAD_TAU = 0.05 
W_CAT = 0.50 
P_MISS = 0.10
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
   - 0.3 ≤ norm_dist < 0.55: Moderate similarity — the hardest zone; examine
     features carefully (accuracy bottoms out near the decision threshold)
   - norm_dist ≥ 0.55: Weak similarity (potentially ABNORMAL)

Decision criteria:
- ABNORMAL if: significant deviation from cluster AND high normalized distance
- NORMAL if: aligns with cluster patterns AND low normalized distance
- Consider both statistical deviation and distance together
- Confidence: 0=uncertain, 100=certain
- Reference specific feature values and normalized distance in reasoning

Respond with ONLY compact JSON (no markdown, no extra text):
{"cls":"NORMAL|ABNORMAL","confidence":<0-100 integer>,"reasoning":"<=25 words citing concrete feature values and nd"}""")
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
    global _MODE, _analyzer_builder, _teacher_builde5r
    assert mode in ("agent", "forward")
    _MODE = mode
    if mode == "agent":
        _analyzer_builder = build_analyzer_user_prompt_agent
        _teacher_builder  = build_teacher_prompt_with_gt_agent
    else:
        _analyzer_builder = build_analyzer_user_prompt_forward
        _teacher_builder  = build_teacher_prompt_with_gt_forward


def build_analyzer_user_prompt(*a, **kw):
    # v2: PROMPT_VERSION 스위치 (set_prompt_version / env PROMPT_VERSION)
    if globals().get("PROMPT_VERSION", "v1") == ("v2", "v3"):
        fn = (build_analyzer_user_prompt_forward_v2 if _MODE == "forward"
              else build_evaluator_user_prompt_v2)
        return fn(*a, **kw)
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

# ════════════════════════════════════════════════════════════════
# PROMPT v2 — prefill 재설계 (multi-cluster memory bank intuition 정합)
# ----------------------------------------------------------------
# 설계 근거 (개념 논의 확정분):
#  ① 이웃 feature 원시 덤프 제거 — 판정 신호는 거리뿐(논문 Eq.7-10),
#     원시 덤프는 소형 모델의 환각 근거로 작용 실증("source 'Ads' deviates"
#     오답: 이웃 3명 전부 Ads). → top-1 합의 다이제스트 1줄로 대체.
#  ② 원시 mean/std 표 제거 — 소형 모델에 암산 요구 대신 z를 사전 계산해
#     [ok/DEVIANT] 태그로 제공 (NB04 고유율 18.9→75.8% 개선의 일반화).
#  ③ 정적 instruction 선두 배치 — llama.cpp prefix KV cache 적중을 위해
#     [정적 instruction][가변 TEST/FIT/RETRIEVAL] 순서. per-request prefill
#     ~350→~150 tok.
#  ④ what_missing 반구조화 — "<direction>: <evidence>" (direction은
#     planner/지표가 기계 소비, evidence는 아래 표에서만 인용).
# train-serve: v2는 재생성-재학습 사이클과 한 몸 — PROMPT_VERSION으로
# v1(현행 어댑터)과 공존. 서빙은 새 어댑터 배포 시점에 env로 전환.
# ════════════════════════════════════════════════════════════════

def set_prompt_version(v: str):
    global PROMPT_VERSION
    PROMPT_VERSION = str(v).lower()

# ── agent 출력 키 스위치: "1key"(S4, 서빙 기본) | "3key"(S3 재현) ──
#   train-serve 정합: 서빙(local_analyzer)은 1key 고정 — 3key는 오프라인 실험 전용.

def set_agent_keys(k: str):
    global AGENT_OUTPUT_KEYS
    assert k in ("1key", "3key")
    AGENT_OUTPUT_KEYS = k

# ── [S3] 3-key evaluator instruction — model이 cls/sufficiency/what_missing 전부 출력 ──
#   (Stage 3 재현용. sufficiency는 [SYSTEM-COMPUTED] 복사, ambiguous cls는 evidence 판단)
EVALUATOR_INSTRUCTION_V2_3KEY = """Output cls, sufficiency, and (only if insufficient) what_missing.
- cls: if confident, follow nd (nd≲0.3→NORMAL, nd≳0.55→ABNORMAL; band may shift);
  if ambiguous, nd is unreliable — weigh the CLUSTER FIT table + retrieval.
- sufficiency: copy the [SYSTEM-COMPUTED] value.
- what_missing: null unless ambiguous; judges RETRIEVAL quality, not the class. "<direction>: <evidence>":
    cluster_mismatch      — a numeric [DEVIANT] FIT feature (name+z) → reroute
    neighbor_insufficient — name 1-2 features under 'differs' + the neighbor_spread state → expand
  Cite from the tables above; never restate the distance.
Output ONLY compact JSON:
{"cls":"NORMAL|ABNORMAL","sufficiency":"confident|ambiguous","what_missing":"<direction>: <evidence>" or null}"""

# ── [S4] 1-key evaluator instruction — cls/sufficiency는 코드(nd)가 결정, model은 wm만 ──
EVALUATOR_INSTRUCTION_V2 = """Judge the RETRIEVAL quality below and state what is missing.
Format "<direction>: <evidence>", direction is exactly one of:
  cluster_mismatch      — a [DEVIANT] FIT feature (name+z) → the assigned cluster is wrong → reroute
  neighbor_insufficient — 1-2 features under 'differs' + the neighbor_spread state → expand
Cite 1-2 features with values from the tables above; never restate the distance.
Output ONLY compact JSON:
{"what_missing":"<direction>: <evidence>"}"""

# ── [S5/V3] typed 1-key — v2의 "<direction>: <evidence>" 형식에 accept를 더한
#   3-way. cls/sufficiency는 v2와 동일하게 코드(nd)가 결정 — 모델 자유 출력은
#   여전히 what_missing 1-key뿐이고, direction 어휘만 {cm, ni, accept}로 교체.
#   (cm=cluster_mismatch, ni=neighbor_insufficient의 축약 — S5 학습 토큰 절감)

# EVALUATOR_INSTRUCTION_V3 = """Judge the RETRIEVAL quality below and state what is missing.
# Format "<direction>: <evidence> — <rationale>", direction is exactly one of:
#   cluster_mismatch     — a FIT feature tagged [DEVIANT], or a PERSONA DOMINANCE feature
#            tagged [RARE] (name+value) → this transaction may be routed to the wrong
#            persona cluster (NOT itself a fraud signal; triggers re-routing)
#   neighbor_insufficient — RETRIEVAL shows [spread] AND at least 2 features under
#            'differs'. Cite the differing features with the HIGHEST dom(...) value
#            in PERSONA DOMINANCE, at most 2.
#   accept — no [DEVIANT] and no [RARE], and neighbors are [dense] or fewer than 2
#            features differ → no actionable retrieval defect. Cite an in-range
#            feature (name+z, e.g. 'purchase_value z=+0.3') or a [match] feature.
# evidence: cite 1-3 feature names with values from the tables above; never restate 
# the distance. Cite ALL features that qualify, up to 3.
# Output ONLY compact JSON:
# {"what_missing":"<direction>: <evidence> — <rationale>"}"""

EVALUATOR_INSTRUCTION_V3 = """Judge the RETRIEVAL quality below and state what is missing.
Format "<direction>: <evidence> — <rationale>", direction is exactly one of:
  cluster_mismatch     — a FIT feature tagged [DEVIANT], or a PERSONA DOMINANCE feature
           tagged [RARE] (name+value) → this transaction may be routed to the wrong
           persona cluster (NOT itself a fraud signal; triggers re-routing)
  neighbor_insufficient — RETRIEVAL shows [spread] AND at least 2 features under
           'differs'. Cite the differing features with the HIGHEST dom(...) value
           in PERSONA DOMINANCE, at most 2.
  accept — no [DEVIANT] and no [RARE], and neighbors are [dense] or fewer than 2
           features differ. Cite ONE in-range numeric (name+z) AND, when a
           PERSONA DOMINANCE feature is tagged [match], cite that too.
evidence: cite 1-3 feature names with values from the tables above; never restate 
the distance. Cite ALL features that qualify, up to 3.
Output ONLY compact JSON:
{"what_missing":"<direction>: <evidence> — <rationale>"}"""

FORWARD_INSTRUCTION_V2 = """Classify the transaction as NORMAL or ABNORMAL using the system-computed evidence below.
- cls: follow nd (nd≲0.3→NORMAL, nd≳0.55→ABNORMAL; band may shift).
- reasoning: cite THE SINGLE MOST DECISIVE feature from the tables below with its value/z
  (nd may be secondary). <=25 words. Never vague phrases like "close to cluster range"
  without a named value.
- confidence: 0-100 integer reflecting how decisive the evidence is.

Output ONLY compact JSON (no markdown):
{"cls":"NORMAL|ABNORMAL","confidence":<0-100 integer>,"reasoning":"..."}"""

def format_cluster_fit(test_features: Dict, persona: Optional[Dict],
                       z_tau: float = None) -> str:
    """수치형 z 사전계산표 — persona가 numeric 한정인 결정과 정합.
    (categorical 적합 증거는 format_retrieval_digest의 합의 라인이 담당)"""
    z_tau = Z_TAU if z_tau is None else z_tau
    nst = (persona or {}).get("numeric_stats") or {}
    lines = []
    for col in NUMERIC_COLS:
        s = nst.get(col)
        v = test_features.get(col)
        if not s or v is None:
            continue
        sd = float(s.get("std", 0.0)) or 1e-6
        z = (float(v) - float(s.get("mean", 0.0))) / sd
        tag = "DEVIANT" if abs(z) >= z_tau else "ok"
        lines.append(f"  - {col}: value={round(float(v), 2)}, z={z:+.2f} [{tag}]")
    return "\n".join(lines) if lines else "  (no cluster profile)"

# ── [v4-compact] 동적 블록 압축 포맷터 ──────────────────────────────
#   instruction 은 건드리지 않는다. 여기만 줄여야 실제 prefill 이 줄어든다.

def format_transaction_compact(data: Dict) -> str:
    """섹션 헤더/불릿/공백줄 제거. 값은 그대로 보존."""
    cols = ["purchase_value", "age", "sex", "source", "browser",
            "weekday_purchase", "month_purchase", "IP_country"]
    out = []
    for c in cols:
        v = data.get(c)
        if v is None:
            continue
        out.append(f"{c}={round(v, 2) if isinstance(v, float) else v}")
    return "\n".join(out)


def format_cluster_fit_compact(test_features: Dict, persona: Optional[Dict],
                               z_tau: float = None) -> str:
    """value= 제거 — TEST 블록과 중복. z 와 태그는 유지(instruction 이 참조)."""
    z_tau = Z_TAU if z_tau is None else z_tau
    nst = (persona or {}).get("numeric_stats") or {}
    lines = []
    for col in NUMERIC_COLS:
        s, v = nst.get(col), test_features.get(col)
        if not s or v is None:
            continue
        sd = float(s.get("std", 0.0)) or 1e-6
        z = (float(v) - float(s.get("mean", 0.0))) / sd
        lines.append(f"{col} z={z:+.2f} [{'DEVIANT' if abs(z) >= z_tau else 'ok'}]")
    return "\n".join(lines) if lines else "(no cluster profile)"


def _retrieval_differs(distances, test_features, topk_feats) -> set:
    """PERSONA 압축에서 '남길 행'을 정하기 위해 differs 집합만 뽑는다."""
    feats = [f for f in (topk_feats or []) if f]
    dif = set()
    for c in CATEGORICAL_COLS:
        tv = test_features.get(c)
        if tv is None:
            continue
        votes = [f.get(c) for f in feats if f.get(c) is not None]
        if not votes:
            continue
        if sum(str(tv) == str(v) for v in votes) * 2 < len(votes):
            dif.add(c)
    return dif


def format_persona_dominance_compact(test_features: Dict, persona: Optional[Dict],
                                     keep: Optional[set] = None) -> str:
    """★ dominant '값'과 빈도를 모두 유지한다 (safe/mid).
       V3 instruction 이 'HIGHEST dom(...) value' 로 ni 축을 랭킹하므로
       값을 지우면 ni 가 선택되지 않는다(실측: ni=0/1000).
       mid 이상에서만 'ok' 행을 differs 기준으로 필터링한다."""
    node = (persona or {}).get("categorical_distribution") or {}
    _drop_val = (PROMPT_COMPACT == "aggressive")
    _filter_ok = PROMPT_COMPACT in ("mid", "aggressive")
    rows = []
    for c in CATEGORICAL_COLS:
        nd_ = node.get(c)
        if not nd_:
            continue
        fr = float(nd_.get("frequency", 0.0))
        dist = nd_.get("distribution") or {}
        tot = sum(float(v) for v in dist.values()) or 1.0
        p = float(dist.get(str(test_features.get(c)), 0)) / tot
        tag = ("RARE" if (fr >= W_CAT and p < P_MISS)
               else "match" if str(test_features.get(c)) == str(nd_.get("dominant"))
               else "ok")
        if tag == "ok" and _filter_ok and keep is not None and c not in keep:
            continue
        if _drop_val:
            rows.append(f"{c} dom={fr:.2f} [{tag}]")
        else:
            rows.append(f"  - {c}: dom={nd_.get('dominant')}({fr:.2f}) [{tag}]")
    return "\n".join(rows) if rows else "  (no categorical profile)"


def format_retrieval_digest_compact(distances, test_features,
                                    topk_feats=None, spread_tau=None) -> str:
    """★ nd(top1) 만 제거한다 (instruction: never restate the distance).
       neighbor_spread 수치는 ni 게이트의 1차 증거이므로 반드시 유지 —
       태그만 남기면 neighbor_insufficient 가 0 으로 붕괴한다(실측)."""
    if not distances:
        return "No retrieved results."
    d = sorted(float(x) for x in distances if x != float("inf"))
    spread_tau = NI_SPREAD_TAU if spread_tau is None else spread_tau
    line1 = ""
    if len(d) >= 3 and d[0] > 0:
        sp = float(np.std(d[:3]))
        line1 = f"neighbor_spread={sp:.3f} [{'spread' if sp >= spread_tau else 'dense'}]"
    elif PROMPT_COMPACT == "aggressive":
        line1 = "[dense]"
    lines = [line1] if line1 else []
    feats = [f for f in (topk_feats or []) if f]
    if feats:
        agr, dif = [], []
        for c in CATEGORICAL_COLS:
            tv = test_features.get(c)
            if tv is None:
                continue
            votes = [f.get(c) for f in feats if f.get(c) is not None]
            if not votes:
                continue
            (agr if sum(str(tv) == str(v) for v in votes) * 2 >= len(votes)
             else dif).append(c)
        lines.append(f"agrees: {', '.join(agr) or '(none)'} | "
                     f"differs: {', '.join(dif) or '(none)'}")
    else:
        lines.append("agrees: (unavailable) | differs: (unavailable)")
    return "\n".join(lines)

def format_retrieval_digest(distances, test_features, topk_feats=None, spread_tau=None):
    if not distances: return "No retrieved results."
    d = sorted(float(x) for x in distances if x != float("inf"))
    line1 = f"nd(top1)={normalize_dist(d[0]):.2f}"
    spread_tau = NI_SPREAD_TAU if spread_tau is None else spread_tau
    if len(d) >= 3 and d[0] > 0:
        # pl = (d[min(4,len(d)-1)] - d[0]) / d[0]
        # line1 += f" | neighbor_spread={pl:.2f} [{'dense' if pl<=spread_tau else 'spread'}]"
        sp = float(np.std(d[:3])) if len(d)>=2 else 0.0
        line1 += f" | neighbor_spread={sp:.3f} [{'spread' if sp>=spread_tau else 'dense'}]"    
    lines = [line1]
    feats = [f for f in (topk_feats or []) if f]
    if feats:
        # k = len(feats)
        agr, dif = [], []
        for c in CATEGORICAL_COLS:
            tv = test_features.get(c)
            if tv is None: continue
            votes = [f.get(c) for f in feats if f.get(c) is not None]
            if not votes: continue
            nm = sum(str(tv) == str(v) for v in votes)
            (agr if nm*2 >= len(votes) else dif).append(c)
        lines.append(f"agrees: {', '.join(agr) or '(none)'} | differs: {', '.join(dif) or '(none)'}")
        # lines.append(f"top-{k} agrees: {', '.join(agr) or '(none)'} | differs: {', '.join(dif) or '(none)'}")
    else:
        lines.append("agrees: (unavailable) | differs: (unavailable)")
    return "\n".join(lines)

def format_persona_dominance(test_features: Dict, persona: Optional[Dict]) -> str:
    """GT의 categorical 게이트/순위 키를 프롬프트에 노출.
       cm(cat) : frequency >= W_CAT AND p(value) < P_MISS
       ni      : frequency 내림차순이 axes 우선순위
       accept  : value == dominant"""
    node = (persona or {}).get("categorical_distribution") or {}
    rows = []
    for c in CATEGORICAL_COLS:
        nd_ = node.get(c)
        if not nd_:
            continue
        fr = float(nd_.get("frequency", 0.0))
        dist = nd_.get("distribution") or {}
        tot = sum(float(v) for v in dist.values()) or 1.0
        p = float(dist.get(str(test_features.get(c)), 0)) / tot
        tag = ("RARE" if (fr >= W_CAT and p < P_MISS)
               else "match" if str(test_features.get(c)) == str(nd_.get("dominant"))
               else "ok")
        # rows.append(f"  - {c}: dom={nd_.get('dominant')}({fr:.2f}), test_p={p:.2f} [{tag}]")
        rows.append(f"  - {c}: dom={nd_.get('dominant')}({fr:.2f}) [{tag}]")
    return "\n".join(rows) if rows else "  (no categorical profile)"

# ── ⑥ : _build_v2 ────────────────────────────────────────────────────

def _build_v2(instruction, test_features, cluster_id, distances, persona, topk_feats):
    _v3 = (PROMPT_VERSION == "v3")
    fit_hdr = ("## CLUSTER FIT (z vs persona)" if _v3
               else f"## CLUSTER #{cluster_id} FIT (z vs profile)")

    # ★ [v4-compact] instruction 은 항상 원문 유지(KV 캐시 영역, 이득 0/리스크 有).
    #   동적 블록만 교체한다.
    if _COMPACT_ON:
        _keep = _retrieval_differs(distances, test_features, topk_feats)
        # ★ CLUSTER FIT 의 value= 제거는 mid 이상에서만 (TEST 와 중복이지만
        #   모델이 'name+value' 인용을 학습했을 수 있어 safe 에서는 보존)
        _fit = (format_cluster_fit_compact(test_features, persona)
                if PROMPT_COMPACT in ("mid", "aggressive")
                else format_cluster_fit(test_features, persona))
        return (
            f"{instruction}\n\n"
            f"## TEST\n{format_transaction_compact(test_features)}\n\n"
            f"{fit_hdr}\n{_fit}\n\n"
            f"## PERSONA DOMINANCE (categorical)\n"
            f"{format_persona_dominance_compact(test_features, persona, keep=_keep)}\n\n"
            f"## RETRIEVAL\n"
            f"{format_retrieval_digest_compact(distances, test_features, topk_feats)}"
        )
    # if PROMPT_COMPACT:
    #     _keep = _retrieval_differs(distances, test_features, topk_feats)
    #     return (
    #         f"{instruction}\n\n"
    #         f"## TEST\n{format_transaction_compact(test_features)}\n\n"
    #         f"{fit_hdr}\n"
    #         f"{format_cluster_fit_compact(test_features, persona)}\n\n"
    #         f"## PERSONA DOMINANCE (categorical)\n"
    #         f"{format_persona_dominance_compact(test_features, persona, keep=_keep)}\n\n"
    #         f"## RETRIEVAL\n"
    #         f"{format_retrieval_digest_compact(distances, test_features, topk_feats)}"
    #     )

    return (
        f"{instruction}\n\n"
        f"## TEST\n{format_transaction(test_features)}\n\n"
        f"{fit_hdr}\n"
        f"{format_cluster_fit(test_features, persona)}\n\n"
        f"## PERSONA DOMINANCE (categorical)\n"
        f"{format_persona_dominance(test_features, persona)}\n\n"
        f"## RETRIEVAL\n"
        f"{format_retrieval_digest(distances, test_features, topk_feats)}"
    )

def build_evaluator_user_prompt_v2(test_features: Dict, cluster_id,
                                   top5_orig_indices: List[int],
                                   distances: List[float],
                                   persona: Optional[Dict] = None,
                                   topk_feats: Optional[Dict] = None) -> str:
    """오프라인(NB): coreset_feature_map으로 top-1 feature 자동 해석.
    서빙: topk_feats를 직접 전달(local_analyzer가 retriever 결과에서 추출)."""
    if topk_feats is None and top5_orig_indices:
        try:
            topk_feats = [coreset_feature_map.get(int(i)) for i in top5_orig_indices[:3]]
        except (NameError, TypeError):
            topk_feats = None
    _instr = (EVALUATOR_INSTRUCTION_V3 if PROMPT_VERSION == "v3"   # ★ S5 typed 1-key
              else EVALUATOR_INSTRUCTION_V2_3KEY if AGENT_OUTPUT_KEYS == "3key"
              else EVALUATOR_INSTRUCTION_V2)          # ★ S3/S4 스위치
    return _build_v2(_instr, test_features, cluster_id,
                     distances, persona, topk_feats)

def build_analyzer_user_prompt_forward_v2(test_features: Dict, cluster_id,
                                          top5_orig_indices: List[int],
                                          distances: List[float],
                                          persona: Optional[Dict] = None,
                                          topk_feats: Optional[Dict] = None) -> str:
    if topk_feats is None and top5_orig_indices:
        try:
            topk_feats = [coreset_feature_map.get(int(i)) for i in top5_orig_indices[:3]]
        except (NameError, TypeError):
            topk_feats = None
    return _build_v2(FORWARD_INSTRUCTION_V2, test_features, cluster_id,
                     distances, persona, topk_feats)
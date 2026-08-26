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


def compute_evidence_axes(test_features: Dict, persona: Optional[Dict],
                          distances: Optional[List[float]] = None,
                          topk_feats: Optional[List[Dict]] = None,
                          z_tau: float = None, spread_tau: float = None) -> Dict:
    """
    ★ [신규, 추가 전용] format_retrieval_digest/format_cluster_fit/format_
    persona_dominance_compact가 "프롬프트 텍스트"로만 만들던 agrees/differs/
    deviant/rare/spread_tag를 구조화된 dict로 반환한다.

    기존 포맷터 함수는 단 한 줄도 수정하지 않았다 — 이 함수는 동일한 수식을
    독립적으로 한 번 더 계산하는 새 함수이므로, 프롬프트 출력(따라서 SFT된
    모델의 train-serve 정합성)에는 어떤 영향도 없다. local_analyzer.py가
    AnalyzeResponse에 이 결과를 실어 controller.py로 넘기기 위해 만들었다
    (analyzer_finetune 노트북들이 참조하는 기존 5개 함수는 그대로 둔 채,
    소비자만 하나 늘리는 방식).

    Returns:
        {"agrees": [...], "differs": [...], "deviant": [...], "rare": [...],
         "persona_match": [...], "spread_tag": "spread"|"dense"|None}
    """
    z_tau = Z_TAU if z_tau is None else z_tau
    spread_tau = NI_SPREAD_TAU if spread_tau is None else spread_tau
    persona = persona or {}
    feats = [f for f in (topk_feats or []) if f]

    # agrees/differs — format_retrieval_digest의 categorical 다수결과 동일 수식
    agrees, differs = [], []
    for c in CATEGORICAL_COLS:
        tv = test_features.get(c)
        if tv is None:
            continue
        votes = [f.get(c) for f in feats if f.get(c) is not None]
        if not votes:
            continue
        nm = sum(str(tv) == str(v) for v in votes)
        (agrees if nm * 2 >= len(votes) else differs).append(c)

    # deviant — format_cluster_fit의 numeric z-score와 동일 수식
    nst = persona.get("numeric_stats") or {}
    deviant = []
    for c in NUMERIC_COLS:
        s, v = nst.get(c), test_features.get(c)
        if not s or v is None:
            continue
        sd = float(s.get("std", 0.0)) or 1e-6
        z = (float(v) - float(s.get("mean", 0.0))) / sd
        if abs(z) >= z_tau:
            deviant.append(c)

    # rare / persona_match — format_persona_dominance_compact의 dominant/RARE와 동일 수식
    node = persona.get("categorical_distribution") or {}
    rare, persona_match = [], []
    for c in CATEGORICAL_COLS:
        nd_ = node.get(c)
        if not nd_:
            continue
        fr = float(nd_.get("frequency", 0.0))
        dist = nd_.get("distribution") or {}
        tot = sum(float(v) for v in dist.values()) or 1.0
        tv = test_features.get(c)
        p = float(dist.get(str(tv), 0)) / tot
        if fr >= W_CAT and p < P_MISS:
            rare.append(c)
        if str(tv) == str(nd_.get("dominant")):
            persona_match.append(c)

    # spread_tag — format_retrieval_digest의 neighbor_spread와 동일 수식
    spread_tag = None
    ds = sorted(float(x) for x in (distances or []) if x != float("inf"))
    if len(ds) >= 3 and ds[0] > 0:
        sp = float(np.std(ds[:3]))
        spread_tag = "spread" if sp >= spread_tau else "dense"

    return {"agrees": agrees, "differs": differs, "deviant": deviant,
            "rare": rare, "persona_match": persona_match, "spread_tag": spread_tag}

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


# ============================================================
# ── Planner (GRPO) User Prompt — evaluator와 별개 네임스페이스 ──
# ============================================================
# ★ [신규] evaluator(build_analyzer_user_prompt*/EVALUATOR_INSTRUCTION_*)와
# 같은 서빙 파이프라인(controller.py)에 붙지만 역할이 다르다 — evaluator는
# state 해석(SFT 전용, §12 evidence interpreter), planner는 action 선택
# (GRPO, §12 counterfactual decision). 그래서 별도 SYSTEM/INSTRUCTION/build_*
# 세트로 완전히 분리해 evaluator 쪽 prompt/SFT에는 한 글자도 영향을 주지
# 않는다.
#
# ★ PROMPT_COMPACT을 참조하지 않는다(의도적) — 직전 turn에서 확인된
# evaluator 버그(neighbor_insufficient 실종)의 원인 중 하나가 compact
# 프롬프트 전환 과정에 있었다. planner는 먼저 이 non-compact 버전으로
# SFT/GRPO의 최고 성능선을 확인한 뒤, 필요하면 별도 compact 버전을
# 만들어 A/B한다 — evaluator처럼 "나중에 compact화했다가 회귀 원인을
# 못 찾는" 순서를 반복하지 않기 위함.

PLANNER_SYSTEM_PROMPT = (
    "You are a retrieval planner for a fraud-detection agent. Given a "
    "transaction's evidence (already retrieved from its assigned cluster), "
    "decide whether additional retrieval is worth its cost, and if so, which "
    "kind. Re-search is not free — most transactions need no further action."
)

PLANNER_INSTRUCTION_V1 = """Decide the retrieval action for this transaction.

## AVAILABLE ACTIONS (choose exactly one)
  stop            - current evidence is enough; do not search again. cost=0.
  global_knn      - search the ENTIRE normal memory bank (ignore cluster/features)
                    for a possibly closer normal neighbor. Use when the CLUSTER FIT
                    table shows a [DEVIANT] feature (assigned cluster may be wrong)
                    AND no PERSONA DOMINANCE feature is [RARE] (a rare value would
                    make a feature-targeted search more precise than a blind one).
                    cost=1.5.
  contextual_knn  - search using the SAME embedding but softly bias toward
                    neighbors that match up to 2 of the CANDIDATE FEATURES below
                    (a soft boost, not a hard filter - embedding similarity is
                    never discarded). Use when RETRIEVAL shows [spread] or a
                    candidate feature is tagged [RARE]/[match] - i.e. the original
                    feature context can narrow the search better than embedding
                    distance alone. cost=1.2.

## CANDIDATE FEATURES (for contextual_knn only - pick at most 2, ONLY from this list)
{candidate_features_line}

## DECISION RULE
Prefer stop unless the evidence above concretely points to a retrieval defect
(cluster mismatch or neighbor insufficiency) that one of the other two actions is
specifically suited to fix. Being merely uncertain is not sufficient grounds to
search again - name the concrete evidence (a [DEVIANT]/[RARE]/[spread] tag) that
justifies the extra cost.

Output ONLY compact JSON, no markdown:
{{"action":"stop"}}
{{"action":"global_knn"}}
{{"action":"contextual_knn","features":["<field1>","<field2 optional>"]}}"""

# ============================================================
# ── Planner Prompt v2 — evidence-only (DECISION RULE 휴리스틱 제거) ──
# ============================================================
# ★ [실험] PLANNER_INSTRUCTION_V1의 "Use when [DEVIANT]이고 [RARE]가 아니면
# global_knn을 써라" 같은 서술은 제가 상식적으로 쓴 휴리스틱이지, 실제 정답
# 라벨(oracle_best의 argmax reward)이 이 규칙을 따라 계산된 게 아니다 —
# 정답은 순수하게 Δmargin과 λ·cost의 숫자 비교로만 정해진다. 프롬프트가
# "이렇게 판단하라"고 말하는 서술과, 실제로 맞혀야 하는 정답의 계산 방식이
# 다른 스토리를 말하고 있어서 오히려 학습을 헷갈리게 할 수 있다는 가설을
# 검증하기 위한 버전. TEST/CLUSTER FIT/PERSONA DOMINANCE/RETRIEVAL(evidence
# 블록)은 그대로 유지 — 이건 "서술적 지시"가 아니라 "관찰 가능한 사실"이라
# 그대로 둔다.

PLANNER_INSTRUCTION_V2 = """Choose exactly one retrieval action for this transaction, based on the evidence below.

## ACTIONS
  stop            - cost=0
  global_knn      - cost=1.5
  contextual_knn  - cost=1.2, requires "features": up to 2 values from CANDIDATE FEATURES

## CANDIDATE FEATURES (for contextual_knn only)
{candidate_features_line}

Output ONLY compact JSON, no markdown, no explanation:
{{"action":"stop"}}
{{"action":"global_knn"}}
{{"action":"contextual_knn","features":["<field1>","<field2 optional>"]}}"""

# ============================================================
# ── Planner Prompt v3 — 제약 디코딩(Constrained Decoding) 전용 ──
# ============================================================
# ★ V2와의 차이는 "무엇을 뺐는가"가 아니라 "무엇을 왜 남겼는가"다.
#
# 뺀 것 (action 선택을 편향시키는 서술):
#   - DECISION RULE 블록 (V1) — 정답은 Δmargin과 λ·cost의 숫자 비교로만
#     정해지는데, 프롬프트가 다른 판단 기준을 말하면 학습 신호와 충돌한다.
#   - "Use when [DEVIANT]..." 류 휴리스틱 (V1)
#   - "Output ONLY compact JSON, no markdown, no explanation" (V1/V2)
#     — 제약 디코딩이 이미 구조적으로 보장하므로 훈계가 무의미하다.
#
# 남긴 것 (출력 '모양'의 서술):
#   - JSON 스키마 예시. 제약 디코딩은 허용 토큰 위에서 확률을 재정규화하는데,
#     base가 유효 포맷에 거의 확률을 주지 않으면 재정규화 결과가 사실상
#     임의값이 되어 policy gradient가 노이즈가 된다. 포맷 설명을 남겨두면
#     확률질량이 유효 집합 근처에 머물러 제약이 '폭력적 사영'이 아니라
#     '가벼운 손질'이 된다. → 포맷 정의는 손해가 아니라 필요조건이다.
#   - evidence 블록 4개 (관찰 가능한 사실이지 지시가 아님)
#
# cost/최대 feature 수를 하드코딩하지 않는다 — agent_tools의 registry가
# 단일 출처이고(binary cost로 변경됨), 프롬프트가 옛 값(1.2/1.5)을 말하면
# reward와 프롬프트가 다른 이야기를 하게 된다.
PLANNER_INSTRUCTION_V3 = """Choose exactly one retrieval action for this transaction, based on the evidence below.

## ACTIONS
{action_lines}

## CANDIDATE FEATURES (pick at most {max_features}, ONLY from this list)
{candidate_features_line}

## OUTPUT SHAPE
{{"action":"stop"}}
{{"action":"contextual_knn","features":["source"]}}
{{"action":"attribute_knn","features":["sex","source"]}}"""

def build_planner_action_lines(
    actions: Optional[List[str]] = None,
    costs: Optional[Dict[str, float]] = None,
) -> str:
    """ACTIONS 블록을 호출자가 준 registry 정보로 만든다.
    prompts.py가 agent_tools를 import하면 순환 의존이 되므로, 값은 항상
    바깥(노트북/local_planner)에서 주입받는다."""
    acts = actions or ["stop", "contextual_knn"]
    cs = costs or {}
    lines = []
    for a in acts:
        c = cs.get(a, 0.0 if a == "stop" else 1.0)
        if a == "stop":
            lines.append(f"  stop            - no further retrieval. cost={c:g}")
        else:
            lines.append(f'  {a:<15s} - requires "features". cost={c:g}')
    return "\n".join(lines)

def build_planner_user_prompt_v2(
    test_features: Dict,
    cluster_id,
    distances: List[float],
    persona: Optional[Dict] = None,
    topk_feats: Optional[List[Dict]] = None,
    candidate_features: Optional[List[str]] = None,
) -> str:
    """build_planner_user_prompt()와 evidence 블록(4개)은 완전히 동일 --
    instruction만 PLANNER_INSTRUCTION_V2(action 설명/cost만, "언제 써라"
    서술과 DECISION RULE 없음)로 교체."""
    cf = candidate_features or []
    cf_line = ", ".join(cf) if cf else "(none - contextual_knn is not selectable this turn)"
    instr = PLANNER_INSTRUCTION_V2.format(candidate_features_line=cf_line)
    fit_hdr = "## CLUSTER FIT (z vs persona)"
    return (
        f"{instr}\n\n"
        f"## TEST\n{format_transaction(test_features)}\n\n"
        f"{fit_hdr}\n{format_cluster_fit(test_features, persona)}\n\n"
        f"## PERSONA DOMINANCE (categorical)\n"
        f"{format_persona_dominance(test_features, persona)}\n\n"
        f"## RETRIEVAL\n"
        f"{format_retrieval_digest(distances, test_features, topk_feats)}"
    )

def build_planner_user_prompt_v3(
    test_features: Dict,
    cluster_id,
    distances: List[float],
    persona: Optional[Dict] = None,
    topk_feats: Optional[List[Dict]] = None,
    candidate_features: Optional[List[str]] = None,
    actions: Optional[List[str]] = None,
    action_costs: Optional[Dict[str, float]] = None,
    max_features: int = 2,
) -> str:
    """제약 디코딩 전용. evidence 블록 4개는 V1/V2와 완전히 동일하고
    instruction만 교체된다 — 프롬프트 버전 간 비교에서 evidence가 교란변수가
    되지 않도록 하기 위함.

    actions / action_costs / max_features는 agent_tools registry에서 받아
    넘긴다. 넘기지 않으면 stop/contextual_knn + binary cost로 degrade한다.
    """
    cf = candidate_features or []
    cf_line = ", ".join(cf) if cf else "(none - only stop is selectable this turn)"
    instr = PLANNER_INSTRUCTION_V3.format(
        action_lines=build_planner_action_lines(actions, action_costs),
        candidate_features_line=cf_line,
        max_features=max_features,
    )
    fit_hdr = "## CLUSTER FIT (z vs persona)"
    return (
        f"{instr}\n\n"
        f"## TEST\n{format_transaction(test_features)}\n\n"
        f"{fit_hdr}\n{format_cluster_fit(test_features, persona)}\n\n"
        f"## PERSONA DOMINANCE (categorical)\n"
        f"{format_persona_dominance(test_features, persona)}\n\n"
        f"## RETRIEVAL\n"
        f"{format_retrieval_digest(distances, test_features, topk_feats)}"
    )

def build_planner_user_prompt(
    test_features: Dict,
    cluster_id,
    distances: List[float],
    persona: Optional[Dict] = None,
    topk_feats: Optional[List[Dict]] = None,
    candidate_features: Optional[List[str]] = None,
) -> str:
    """
    planner(GRPO) 전용 user prompt. evaluator의 _build_v2와 형태(TEST/CLUSTER
    FIT/PERSONA DOMINANCE/RETRIEVAL 4블록)는 동일하게 재사용하지만 —
      1) 항상 non-compact(전체) 버전만 만든다. PROMPT_COMPACT 분기 자체가
         없다(위 모듈 docstring 참고).
      2) instruction 뒤에 CANDIDATE FEATURES(= agrees∪persona_match, 오라클
         rollout의 build_grpo_action_space와 동일 pool)를 동적으로 삽입한다.

    candidate_features: controller.py의 _evidence_from_analyzer()가 계산하는
    agrees∪persona_match와 정확히 같은 리스트를 넘겨야 한다 — 서빙(planner
    inference)과 학습(이 함수로 만든 오프라인 프롬프트)이 같은 후보 집합을
    보도록 하기 위함(다르면 학습 때 못 보던 feature가 서빙에서 후보로
    나오거나, 그 반대가 되어 train-serve mismatch가 생긴다). 비어 있으면
    모델이 존재하지 않는 feature를 환각하지 않도록 명시적으로 "없음"을
    적는다.
    """
    cf = candidate_features or []
    cf_line = ", ".join(cf) if cf else "(none - contextual_knn is not selectable this turn)"
    instr = PLANNER_INSTRUCTION_V1.format(candidate_features_line=cf_line)
    fit_hdr = "## CLUSTER FIT (z vs persona)"
    return (
        f"{instr}\n\n"
        f"## TEST\n{format_transaction(test_features)}\n\n"
        f"{fit_hdr}\n{format_cluster_fit(test_features, persona)}\n\n"
        f"## PERSONA DOMINANCE (categorical)\n"
        f"{format_persona_dominance(test_features, persona)}\n\n"
        f"## RETRIEVAL\n"
        f"{format_retrieval_digest(distances, test_features, topk_feats)}"
    )

# ============================================================
# ── Planner Prompt v5 — think + 압축 evidence + 진단/처방 ──
# ============================================================
# prompts.py 맨 끝에 그대로 붙여넣는다. v1~v4는 손대지 않는다
# (프롬프트 버전 간 A/B에서 evidence가 교란변수가 되지 않도록).
#
# ★ v3 → v5 변경 3가지와 각각의 실측 근거
#
# 1) think 출력 추가
#    현재 출력은 {tool 2종} × {feature 조합 21} = 43-way 선택이라, 원리적으로
#    43-way 분류기로 완전히 환원된다. 그러면 0.5B LLM을 쓸 이유가 GBDT 대비
#    없다. think를 붙이면 (a) 분류기가 만들 수 없는 산출물이 생기고,
#    (b) "모델이 evidence를 읽었는가"를 grounded rate로 직접 측정할 수 있다.
#    ★ think는 reward에 넣지 않는다 — outcome에 접지되지 않은 process reward는
#      hacking되고(R1-Searcher가 2단계에서 retrieve-reward를 제거한 이유),
#      품질을 자동 채점할 방법도 없다. 측정 지표로만 쓴다.
#
# 2) DIAGNOSIS 블록 (wm_type)
#    evaluator가 진단한 결함(cluster_mismatch / neighbor_insufficient)이 tool
#    선택과 무상관이었다(실측: 세 진단의 tool 분포가 0.19~0.24로 거의 동일).
#    설계 의도가 데이터로 확인되지 않은 상태이고, 원인 후보 중 하나가
#    "프롬프트가 진단과 처방을 연결해주지 않는다"이다.
#    ※ V1의 "Use when [DEVIANT]..." 휴리스틱과 다르다. 그건 상식으로 만든
#      규칙이었지만, 여기 적는 건 tool의 구현상 성질이다 —
#      contextual = dense pool 재정렬(이웃 부족 보강),
#      attribute  = exact term filter(라우팅 교정).
#
# 3) evidence 압축 + support 노출, TEST 블록 제거
#    · TEST(8개 feature 값) 제거: planner가 정하는 것은 "어느 축으로 좁힐까"이지
#      "값이 무엇인가"가 아니다. eq 필터는 서버가 이 거래의 값으로 자동 채우고,
#      값의 특이성은 [RARE]/[DEVIANT] 태그가 이미 요약한다. 정보 손실이 없다.
#    · support 노출: attribute vs global 이웃 Jaccard가 support 구간에 따라
#      0.29(<1k) ~ 0.68(>25k)로 갈렸다. 선택도가 높으면 두 tool이 사실상 같은
#      검색이 되므로, planner가 이를 보고 골라야 한다.
#    · 0.5B는 프롬프트 길이에 민감하다(evaluator 경험). v3 ~380토큰 → v5 ~200토큰.
#
# ★ stop이 ACTIONS에 없는 이유
#   게이트(검색할지 말지)를 LLM에 맡기는 실험을 3회 했고 전부 실패했다:
#     AUC(p_nonstop) = 0.481 / 0.463 / 0.443  (전부 무정보 이하)
#     같은 입력으로 RandomForest 0.747, |nd-τ| 스칼라 단독 0.68~0.82
#   구조적 원인은 GRPO loss에 있다 — 그룹 6개가 전부 검색이면 reward std=0이라
#   advantage가 0이고 stop을 강화할 gradient가 아예 발생하지 않는다
#   (관측: pred_contextual 85%, silent_group_rate 0.2~0.5).
#   → 게이트는 결정론 임계값(Stage A)으로 빼고 LLM은 tool 선택만 맡는다.
# ============================================================

PLANNER_SYSTEM_PROMPT_V5 = (
    "You are a retrieval planner for a fraud-detection agent. A first retrieval "
    "already ran and an evaluator diagnosed what is missing. Pick the one tool "
    "that repairs that specific defect."
)

# tool의 구현상 성질 — 어떤 결함을 겨냥하는 도구인가
PLANNER_TOOL_ROLE_V5 = {
    # "contextual_knn": "widen to the whole bank, softly prefer matching values "
    #                   "(fixes: neighbors too few or scattered)",
    # "attribute_knn":  "keep only exact feature matches, then take the nearest "
    #                   "(fixes: the assigned cluster looks wrong)",
    "contextual_knn": "widen to the whole bank, softly prefer matching values",
    "attribute_knn":  "keep only exact feature matches, then take the nearest",
    "global_knn":     "search the whole bank with no feature bias",
    "stop":           "no further retrieval",
}

# evaluator 진단 → 한 줄 설명
PLANNER_DIAGNOSIS_HINT_V5 = {
    "cluster_mismatch":      "the assigned cluster looks wrong "
                             "(a feature is [DEVIANT] or [RARE])",
    "neighbor_insufficient": "the retrieved neighbors are too few or too "
                             "scattered to serve as evidence",
    "accept":                "no specific defect was flagged",
}

# 진단 → 우선 처방. 정합률 측정(notebook)의 기준 매핑이다.
# ★ 이건 "정답"이 아니라 "설계 가설"이다. 오라클 정합률이 낮게 나오면
#   가설 자체가 데이터로 반증된 것이고, 프롬프트로는 고칠 수 없다.
PLANNER_DIAGNOSIS_PRESCRIPTION_V5 = {
    "cluster_mismatch":      "attribute_knn",
    "neighbor_insufficient": "contextual_knn",
    "accept":                None,      # 권장 없음 — 정합률 계산에서 제외
}

# think에서 인용할 수 있는 evidence 태그(= grounded 판정 대상)
PLANNER_EVIDENCE_TAGS_V5 = ("DEVIANT", "RARE", "spread", "dense", "match")

PLANNER_INSTRUCTION_V5 = """Pick one retrieval tool to repair the diagnosed defect.

## DIAGNOSIS
{diagnosis_line}

## TOOLS
{action_lines}

## CANDIDATE FEATURES (pick at most {max_features}, ONLY from this list)
{candidate_features_line}

## OUTPUT
Start with the evidence tag in brackets, then the field, then why. Under 20 words.
Example shape (do not copy the words):
{{"think":"[<one tag from EVIDENCE>] <one field from CANDIDATE FEATURES> - <reason>","action":"...","features":[...]}}"""

def build_planner_action_lines_v5(actions: Optional[List[str]] = None) -> str:
    """TOOLS 블록. cost를 쓰지 않는다 — Stage A가 검색 여부를 이미 정했으므로
    이 시점의 비용은 tool 간 동일하고(둘 다 검색 1회), 옛 cost 값(1.2/1.5)을
    노출하면 reward와 프롬프트가 다른 이야기를 하게 된다."""
    acts = actions or ["contextual_knn", "attribute_knn"]
    return "\n".join(f"  {a:<15s} - {PLANNER_TOOL_ROLE_V5.get(a, '')}" for a in acts)


def build_planner_candidate_line_v5(
    candidate_features: Optional[List[str]] = None,
    supports: Optional[Dict[str, int]] = None,
) -> str:
    """★ support(= attribute_knn 시 남는 정상 샘플 수)를 함께 노출한다.
    이론상 임계 s* = k/P (k=top_k, P=contextual pool 크기)를 넘으면 contextual의
    dense pool이 이미 필터 조건을 만족해 두 tool이 같아진다. 실측 임계는
    support ≈ 25,000이었고, 그 위 후보가 전체의 45%였다."""
    cf = candidate_features or []
    if not cf:
        return "(none available)"
    if not supports:
        return ", ".join(cf)
    out = []
    for f in cf:
        s = supports.get(f)
        out.append(f"{f}({s:,})" if isinstance(s, (int, float)) else f)
    return "  ".join(out)


def format_evidence_compact_v5(
    test_features: Dict,
    distances: Optional[List[float]],
    persona: Optional[Dict] = None,
    topk_feats: Optional[List[Dict]] = None,
    z_tau: float = None,
    spread_tau: float = None,
) -> str:
    """v3의 CLUSTER FIT / PERSONA DOMINANCE / RETRIEVAL 3블록(약 250토큰)을
    태그가 붙은 축만 남겨 ~60토큰으로 압축한다.

    ★ 계산식은 format_cluster_fit / format_persona_dominance /
      format_retrieval_digest와 완전히 동일하다(compute_evidence_axes 재사용).
      압축은 '무엇을 빼는가'이지 '어떻게 계산하는가'를 바꾸지 않는다 —
      evaluator에서 compact 전환 후 회귀 원인을 못 찾았던 실수를 반복하지 않기 위함.
    """
    ax = compute_evidence_axes(
        test_features=test_features, persona=persona,
        distances=distances, topk_feats=topk_feats,
        z_tau=z_tau, spread_tau=spread_tau) or {}

    lines = []
    # 1) nd + spread — 게이트/밴드 판정의 근거
    if distances:
        d = sorted(float(x) for x in distances if x != float("inf"))
        if d:
            _sp = float(np.std(d[:3])) if len(d) >= 2 else 0.0
            _st = ax.get("spread_tag") or ""
            lines.append(f"  nd={normalize_dist(d[0]):.2f}  "
                         f"spread={_sp:.3f}" + (f" [{_st}]" if _st else ""))
    # 2) 이웃 합의/불일치 — neighbor_insufficient의 직접 재료
    _ag, _df = ax.get("agrees") or [], ax.get("differs") or []
    lines.append(f"  agrees: {', '.join(_ag) or '(none)'}")
    lines.append(f"  differs: {', '.join(_df) or '(none)'}")
    # 3) 태그가 붙은 축만 — cluster_mismatch의 직접 재료
    for _key, _tag in (("deviant", "DEVIANT"), ("rare", "RARE"),
                       ("persona_match", "match")):
        _v = ax.get(_key) or []
        if _v:
            lines.append(f"  {_tag.lower()}: {', '.join(_v)}")
    return "\n".join(lines)


def build_planner_user_prompt_v5(
    test_features: Dict,
    cluster_id=None,
    distances: Optional[List[float]] = None,
    persona: Optional[Dict] = None,
    topk_feats: Optional[List[Dict]] = None,
    candidate_features: Optional[List[str]] = None,
    actions: Optional[List[str]] = None,
    max_features: int = 2,
    wm_type: Optional[str] = None,
    supports: Optional[Dict[str, int]] = None,
    z_tau: float = None,
    spread_tau: float = None,
) -> str:
    """2-stage(Stage B) 전용 planner 프롬프트. 1-turn.

    v3 대비: TEST 블록 제거, evidence 3블록 압축, DIAGNOSIS/support 추가,
    출력에 think 포함. 약 380토큰 → 약 200토큰.

    cluster_id는 시그니처 호환을 위해 받되 쓰지 않는다 — 프롬프트에 노출해도
    planner가 쓸 정보가 아니고(어느 클러스터인지가 아니라 그 클러스터가
    맞는지가 문제), 토큰만 소모한다.
    """
    _acts = actions or ["contextual_knn", "attribute_knn"]
    _diag = (f"{wm_type}: {PLANNER_DIAGNOSIS_HINT_V5[wm_type]}"
             if wm_type in PLANNER_DIAGNOSIS_HINT_V5 else "(not available)")
    instr = PLANNER_INSTRUCTION_V5.format(
        diagnosis_line=f"  {_diag}",
        action_lines=build_planner_action_lines_v5(_acts),
        candidate_features_line=build_planner_candidate_line_v5(
            candidate_features, supports),
        max_features=max_features,
    )
    ev = format_evidence_compact_v5(test_features, distances, persona,
                                    topk_feats, z_tau, spread_tau)
    return f"{instr}\n\n## EVIDENCE\n{ev}"


# ============================================================
# grounded 판정 — "모델이 evidence를 읽었는가"의 측정 도구
#   think가 (a) 실제로 존재하는 evidence 태그를 인용하고
#          (b) 자기가 고른 feature를 언급했을 때만 grounded로 본다.
#   AND로 묶는 것이 핵심이다 — 태그만 인용하면 앵무새이고,
#   feature만 언급하면 근거가 없다. 둘 다여야 "이 축이 이래서 이 도구"가 된다.
#
#   ★ 이 지표 자체가 목적이 아니다. grounded=True/False 그룹의 achieved 차이가
#     LLM 정당성의 유일한 직접 증거다 — 분류기로는 만들 수 없는 주장.
# ============================================================
def evaluate_think_grounding(
    think: Optional[str],
    chosen_features: Optional[List[str]] = None,
    evidence_axes: Optional[Dict] = None,
) -> Dict:
    """
    Args:
        think          : 모델이 생성한 think 문자열
        chosen_features: 모델이 고른 feature 목록
        evidence_axes  : compute_evidence_axes() 결과 (실제 존재하는 태그 확인용)

    Returns:
        cited_tags     : think가 인용한 태그 중 실제로 존재하는 것
        hallucinated   : think가 인용했으나 실제로는 없는 태그
        field_mentioned: 고른 feature를 think에서 언급했는가
        grounded       : cited_tags가 있고 field_mentioned도 True
    """
    _th = (think or "").lower()
    ax = evidence_axes or {}
    _real = {
        "deviant": bool(ax.get("deviant")),
        "rare":    bool(ax.get("rare")),
        "match":   bool(ax.get("persona_match")),
        "spread":  (ax.get("spread_tag") == "spread"),
        "dense":   (ax.get("spread_tag") == "dense"),
    }
    cited, halluc = [], []
    for t in PLANNER_EVIDENCE_TAGS_V5:
        if t.lower() in _th:
            (cited if _real.get(t.lower(), False) else halluc).append(t)
    _fields = [f for f in (chosen_features or []) if f and f.lower() in _th]
    return {
        "cited_tags": cited,
        "hallucinated_tags": halluc,
        "field_mentioned": bool(_fields),
        "grounded": bool(cited) and bool(_fields),
    }
    
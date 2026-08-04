"""
streamlit_app.py — Fraud Detection Pipeline UI v2 (forwardRAG ⚖ AgentRAG)
==========================================================================
mode-aware 포트폴리오 데모:
  · 응답의 termination/trace로 forward | agent 를 자동 감지해 해당 UI만 렌더
      - forward: 3-key {cls, confidence, reasoning} 카드 + base/analyzer latency
      - agent  : 라운드별 nd/sufficiency/what_missing + PatchCore nd(재검색) +
                 Agent Trace 타임라인 + tool_Δ latency (+ LangGraph mermaid)
  · ⚖ 비교 탭: 같은 세션에서 실행한 forward/agent 결과를 mode별 최신 1건씩
    나란히 비교 (판정/확신도/latency/LLM 호출/라운드/자유 출력)
  · 서버 mode는 orchestrator env(PIPELINE_MODE)가 결정 — UI는 감지·안내만 한다
    (controller/orchestrator의 mode 분기 원칙과 동일)
"""

import json
import os
import time
import requests
import streamlit as st

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

try:
    from streamlit_mermaid import st_mermaid
    MERMAID_OK = True
except ImportError:
    MERMAID_OK = False

# ===========================
# Config (환경변수)
# ===========================

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://anomaly-orchestrator:80")
CONTROLLER_URL   = os.getenv("CONTROLLER_URL",   "http://anomaly-controller:80")
EXPERIMENT_CASE  = os.getenv("EXPERIMENT_CASE",  "pca_64_k100")
ADAPTER_TYPE     = os.getenv("ADAPTER_TYPE",     "mined")

# ===========================
# Helper — 공용
# ===========================

def call_pipeline(payload: dict) -> dict:
    t0 = time.perf_counter()
    resp = requests.post(f"{ORCHESTRATOR_URL}/detect", json=payload, timeout=600)
    resp.raise_for_status()
    out = resp.json()
    out["_client_latency_ms"] = (time.perf_counter() - t0) * 1000
    return out


def detect_mode(result: dict) -> str:
    """응답에서 파이프라인 모드 감지 — orchestrator forward 분기는
    termination='forward' + trace 'forward_analyzer'로 종결된다."""
    if result.get("termination") == "forward":
        return "forward"
    if any("forward_analyzer" in s for s in (result.get("agent_trace") or [])):
        return "forward"
    return "agent"


def _badge(classification: str) -> str:
    return "🔴 ABNORMAL" if classification == "ABNORMAL" else "🟢 NORMAL"


def _mode_badge(mode: str) -> str:
    return "📤 forwardRAG (S2 · 3-key · 전수 LLM)" if mode == "forward" \
        else "🤖 AgentRAG (S4 · 1-key · 선택적 LLM)"


def _llm_called(result: dict, mode: str) -> bool:
    """agent 모드에서 confident-skip이면 LLM 호출 자체가 없다 (비용 축의 핵심)."""
    if mode == "forward":
        return True
    trace = result.get("agent_trace") or []
    return not any("skipped_confident" in s for s in trace)


# ===========================
# Helper — Agent 전용 렌더
# ===========================

_TRACE_ICONS = {
    "pre_router:fast": "⚡", "pre_router:deep": "🔍", "pre_router": "🔍",
    "deep_retrieve": "📥", "retrieve:base_reused": "♻️", "retrieve": "📥",
    "analyze:skipped_confident": "⏭️", "analyze": "🧠", "forward_analyzer": "🧠",
    "plan:expand_within_cluster": "🟡", "plan:reroute_cross_cluster": "🟠",
    "plan:accept_verdict": "🟢",
    "expand_within_cluster:direct": "🔄", "expand_within_cluster:mcp": "🔄",
    "reroute_cross_cluster:direct": "↩️", "reroute_cross_cluster:mcp": "↩️",
    "finalize": "✅",
}


def _render_trace(trace: list) -> None:
    if not trace:
        return
    parts = []
    for step in trace:
        icon = next((v for k, v in _TRACE_ICONS.items() if k in step), "⚪")
        parts.append(f"{icon} `{step}`")
    st.markdown(" → ".join(parts))
    tool_steps = [s for s in trace
                  if "expand_within_cluster" in s or "reroute_cross_cluster" in s]
    if tool_steps:
        st.info(f"🤖 Planner 개입: {' → '.join(tool_steps)}")


def _render_rounds(result: dict) -> None:
    """[agent] 라운드별 nd/sufficiency/what_missing — S4 1-key(det)에서 모델의
    유일한 자유 출력은 wm(감사 주석)이고 cls/suf는 nd가 결정한다.
'final' 엔트리 = 재검색 여부에 따른 최종 nd (raw 또는 PatchCore top-5)."""
    rounds = result.get("agent_rounds") or []
    if not rounds:
        return
    with st.expander("🧭 라운드별 판정 (nd / sufficiency / what_missing)", expanded=True):
        rows = [{
            "round":        r.get("round"),
            "wm_type":      r.get("wm_type"),
            "nd_global":    r.get("nd_global"),
            "flip":         r.get("flip"),
            "contrast":     r.get("contrast_axis"),
            "nd":           r.get("nd", r.get("nd_final")),
            "nd_source":    r.get("nd_source"),
            "reweight_w":   r.get("reweight_w"),
            "research_ms":  r.get("research_ms"),
            "cls":          r.get("cls"),
            "sufficiency":  r.get("sufficiency"),
            "what_missing": r.get("what_missing"),
            "action":       r.get("action"),
        } for r in rounds]
        st.table(rows)
        fin = next((r for r in rounds if r.get("round") == "final"), None)
        if fin and fin.get("nd_round0") is not None and fin.get("nd_final") is not None:
            nd0, ndf = fin["nd_round0"], fin["nd_final"]
            if fin.get("researched"):
                st.success(f"🔀 **재검색 → PatchCore nd**: raw {nd0:.4f} → "
                           f"**{ndf:.4f}** (mode={fin.get('reweight_mode')}, "
                           f"w={fin.get('reweight_w')}, source={fin.get('nd_source')})")
            else:
                st.caption(f"재검색 없음 → raw top-1 nd={ndf:.4f}로 판정")


def _render_latency(latency_ms: dict, mode: str) -> None:
    if not PLOTLY_OK or not isinstance(latency_ms, dict):
        return
    if mode == "forward":
        keys = [k for k in ("base_retrieve", "forward_analyzer")
                if latency_ms.get(k) is not None]
        vals = [latency_ms[k] for k in keys]
        colors = ["#9ECAE9", "#4C78A8"][:len(keys)]
    else:
        base, total = latency_ms.get("base_retrieve"), latency_ms.get("total")
        if base is None or total is None:
            return
        keys = ["base_retrieve", "tool_Δ (analyzer+planner+tool)"]
        vals = [base, total - base]
        colors = ["#9ECAE9", "#F58518"]
    if not keys:
        return
    fig = go.Figure(go.Bar(x=keys, y=vals, text=[f"{v:.0f}ms" for v in vals],
                           textposition="auto", marker_color=colors,
                           marker_line_color="white", marker_line_width=1.5))
    fig.update_layout(title=f"Latency Breakdown — {mode}", height=240,
                      margin=dict(t=36, b=16, l=16, r=16),
                      yaxis_title="ms", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _fetch_mermaid():
    try:
        resp = requests.get(f"{CONTROLLER_URL}/graph/mermaid", timeout=5)
        resp.raise_for_status()
        return resp.json().get("mermaid")
    except Exception:
        return None


# ===========================
# 결과 카드 (mode 분기 렌더)
# ===========================

def render_result(result: dict) -> None:
    mode = detect_mode(result)
    cls, conf = result.get("classification", "UNKNOWN"), result.get("confidence", 0)

    st.markdown(f"##### {_mode_badge(mode)}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("판정", _badge(cls))
    c2.metric("확신도", f"{conf}%",
              help=("모델 confidence (3-key 출력)" if mode == "forward"
                    else "nd 기반 결정론 confidence (det 모드 — 모델 출력 아님)"))
    c3.metric("top-1 distance", f"{result.get('top_1_distance', 0):.4f}",
              help="agent는 재검색 시 PatchCore top-5 nd, 아니면 raw top-1 nd로 판정")
    lat = (result.get("latency_ms") or {}).get("total") or result.get("_client_latency_ms", 0)
    c4.metric("latency", f"{lat/1000:.1f}s")
    # ★ [V3] verify_global 트레이스 — nd_global/flip (legacy 실행 시 "—")
    _rounds_v3 = result.get("agent_rounds") or []
    _ndg = next((r.get("nd_global") for r in reversed(_rounds_v3)
                 if isinstance(r, dict) and r.get("nd_global") is not None), None)
    _flip = any(r.get("flip") for r in _rounds_v3 if isinstance(r, dict))
    c5.metric("nd_global", f"{_ndg:.4f}" if _ndg is not None else "—",
              delta=("flip → NORMAL" if _flip else None),
              help="verify_global(전역 HNSW) nd — cluster 제한 ε 교정값 (v3 모드 전용)")
    _wm = next((r for r in reversed(_rounds_v3)
                if isinstance(r, dict) and r.get("wm_type")), None)
    if _wm:
        st.caption(f"**wm[{_wm['wm_type']}]** "
                   f"{', '.join(_wm.get('wm_axes') or [])} — {_wm.get('wm_rationale') or ''}"
                   + (f"  ·  contrast: {_wm.get('contrast_axis')}"
                      if _wm.get('contrast_axis') else ""))

    if mode == "forward":
        # ── forward: reasoning(모델 자유 출력)이 곧 설명 — agent 전용 UI는 렌더 안 함 ──
        if result.get("reasoning"):
            with st.expander("🧠 Reasoning (모델 3-key 출력)", expanded=True):
                st.write(result["reasoning"])
    else:
        # ── agent: LLM 호출 여부(비용 축) + 라운드 상세 ──
        if _llm_called(result, mode):
            st.caption("💬 ambiguous 밴드 → evaluator LLM 호출됨 (what_missing 감사 주석 생성)")
        else:
            st.caption("⏭️ confident-skip — nd가 밴드 밖이라 **LLM 호출 없이** 결정론 종결 "
                       "(agent 비용 절감의 핵심 경로)")
        _render_rounds(result)

    with st.expander("🔁 Pipeline Trace", expanded=(mode == "agent")):
        _render_trace(result.get("agent_trace", []))

    with st.expander("🔧 파이프라인 상세"):
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"**Primary Cluster**: `{result.get('primary_cluster_id')}`")
            st.markdown(f"**Vec Index**: `{result.get('vec_index')}`")
            st.markdown(f"**Match Type**: `{result.get('match_type')}`")
            st.markdown(f"**Track**: `{result.get('track', '-')}`")
        with m2:
            st.markdown(f"**Termination**: `{result.get('termination', '-')}`")
            if mode == "agent":                      # forward엔 없는 필드 숨김
                st.markdown(f"**Planner Used**: `{result.get('planner_used', '-')}`")
                st.markdown(f"**Tool Mode**: `{result.get('tool_execution_mode', '-')}`")
            st.markdown(f"**Percolate**: `{result.get('percolate_version')}`")
        _render_latency(result.get("latency_ms", {}), mode)


# ===========================
# ⚖ 비교 탭 (mode별 최신 실행 대비)
# ===========================

def render_compare() -> None:
    st.markdown("### ⚖ forwardRAG vs AgentRAG — 같은 세션 실행 비교")
    st.caption(
        "서버 mode는 orchestrator env가 결정합니다. 한쪽 모드로 실행 → 아래 명령으로 "
        "전환 → 같은 입력으로 다시 실행하면 두 결과가 여기에 나란히 잡힙니다.")
    st.code(
        'kubectl patch configmap anomaly-config -n elastic --type merge '
        '--patch \'{"data": {"PIPELINE_MODE": "forward"}}\'\n'
        "kubectl rollout restart deployment/anomaly-orchestrator "
        "deployment/local-analyzer -n elastic",
        language="bash")

    runs = st.session_state.get("runs", {})
    fwd, agt = runs.get("forward"), runs.get("agent")
    if not fwd and not agt:
        st.info("아직 실행 결과가 없습니다 — 단건 추론 탭에서 분석을 실행하세요.")
        return

    def _row(label, f_fn, a_fn, help_=None):
        c0, c1, c2 = st.columns([1.2, 1, 1])
        c0.markdown(f"**{label}**" + (f"  \n<sub>{help_}</sub>" if help_ else ""),
                    unsafe_allow_html=True)
        c1.markdown(f_fn(fwd) if fwd else "—")
        c2.markdown(a_fn(agt) if agt else "—")

    h0, h1, h2 = st.columns([1.2, 1, 1])
    h0.markdown("&nbsp;")
    h1.markdown("#### 📤 forwardRAG (S2)")
    h2.markdown("#### 🤖 AgentRAG (S4)")

    _row("판정",
         lambda r: _badge(r.get("classification", "?")),
         lambda r: _badge(r.get("classification", "?")))
    _row("확신도",
         lambda r: f"{r.get('confidence', '?')}% (모델 3-key)",
         lambda r: f"{r.get('confidence', '?')}% (nd 결정론)",
         help_="forward=모델 출력 / agent=코드 결정 (역할 분리)")
    _row("LLM 호출",
         lambda r: "✅ 전수 (모든 요청)",
         lambda r: ("✅ 호출됨 (ambiguous)" if _llm_called(r, "agent")
                    else "⏭️ **skip** (confident)"),
         help_="agent 가치의 핵심 축 — 선택적 호출 (이론 절감 ≈57%)")
    _row("latency (total)",
         lambda r: f"{(r.get('latency_ms') or {}).get('total', 0)/1000:.1f}s",
         lambda r: f"{(r.get('latency_ms') or {}).get('total', 0)/1000:.1f}s")
    _row("top-1 distance",
         lambda r: f"{r.get('top_1_distance', 0):.4f}",
         lambda r: f"{r.get('top_1_distance', 0):.4f} (재검색 시 patchcore)")
    _row("라운드 / 재검색",
         lambda r: "1 (정적 — 항상 1회)",
         lambda r: str(max(len([x for x in (r.get("agent_rounds") or [])
                                if isinstance(x.get("round"), int)]), 1))
                   + (" + tool" if any("expand" in s or "reroute" in s
                                       for s in (r.get("agent_trace") or [])) else ""))
    _row("모델 자유 출력",
         lambda r: f"reasoning: _{(r.get('reasoning') or '')[:70]}…_",
         lambda r: "what_missing: _" + ((next(
             (x.get("what_missing") for x in (r.get("agent_rounds") or [])
              if x.get("what_missing")), None) or "null (confident)")[:70]) + "_",
         help_="출력 자유도 3-key → 1-key 축소 (판정은 코드로 이관)")

    if PLOTLY_OK and fwd and agt:
        f_lat = (fwd.get("latency_ms") or {}).get("total")
        a_lat = (agt.get("latency_ms") or {}).get("total")
        if f_lat and a_lat:
            fig = go.Figure(go.Bar(
                x=["forwardRAG", "AgentRAG"], y=[f_lat, a_lat],
                text=[f"{f_lat/1000:.1f}s", f"{a_lat/1000:.1f}s"],
                textposition="auto", marker_color=["#4C78A8", "#F58518"]))
            fig.update_layout(title="End-to-end Latency", height=260,
                              margin=dict(t=36, b=16, l=16, r=16), yaxis_title="ms")
            st.plotly_chart(fig, use_container_width=True)


# ===========================
# Streamlit App
# ===========================

st.set_page_config(page_title="Fraud Detection — forwardRAG ⚖ AgentRAG",
                   page_icon="🔍", layout="wide")

st.title("🔍 Fraud Detection Pipeline — forwardRAG ⚖ AgentRAG")
st.caption(
    f"Model: Qwen2.5-0.5B + LoRA ({ADAPTER_TYPE})  |  Experiment: {EXPERIMENT_CASE}  |  "
    f"Orchestrator: {ORCHESTRATOR_URL}  |  서버 mode(PIPELINE_MODE)는 응답에서 자동 감지")

if "runs" not in st.session_state:
    st.session_state["runs"] = {}      # {"forward": 최신 result, "agent": 최신 result}
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# ── 사이드바: 거래 입력 (기존 유지) ──────────────────────────
with st.sidebar:
    st.header("📋 거래 정보 입력")
    purchase_value = st.number_input("purchase_value (z-score)", value=0.0,
                                     step=0.1, format="%.4f")
    age = st.number_input("age (z-score)", value=0.0, step=0.1, format="%.4f")
    sex     = st.selectbox("sex",     ["M", "F"])
    source  = st.selectbox("source",  ["Direct", "Ads", "SEO"])
    browser = st.selectbox("browser", ["Chrome", "IE", "Safari", "FireFox", "Opera"])
    weekday = st.selectbox("weekday_purchase (optional)",
                           ["(없음)", "Monday", "Tuesday", "Wednesday", "Thursday",
                            "Friday", "Saturday", "Sunday"])
    month = st.selectbox("month_purchase (optional)",
                         ["(없음)", "January", "February", "March", "April", "May",
                          "June", "July", "August", "September", "October",
                          "November", "December"])
    ip_country = st.text_input("IP_country (optional)", value="")

    st.divider()
    st.subheader("🔢 Embedding (576-dim)")
    embedding_input = st.text_area("JSON array 형식으로 입력", height=100,
                                   placeholder="[0.123, -0.456, ...]")
    percolate_version = st.selectbox(
        "Percolate Version",
        ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v11", "v12", "v13", "v14"],
        index=11)
    run_btn = st.button("🚀 분석 실행", use_container_width=True, type="primary")

    st.divider()
    if MERMAID_OK:
        if st.toggle("🗺️ Agent Graph 보기 (주석 정본)", value=False):
            # annotated=True(기본): 역할분리(🟠LLM/🔵코드)·evaluator-once·PatchCore nd
            mermaid_txt = _fetch_mermaid()
            if mermaid_txt:
                st_mermaid(mermaid_txt, height=560)
                st.caption("🟠 evaluator LLM(what_missing 1회) · "
                           "🔵 라우팅·planner·판정(코드) · 🟢 종단")
                # mermaid.js 파싱 실패 시 빈 박스만 보이므로, 원문을 fallback으로 노출
                with st.expander("mermaid 원문 (렌더 실패 시 확인)"):
                    st.code(mermaid_txt, language="mermaid")
            else:
                st.warning("Controller `/graph/mermaid` 응답 없음 — controller pod 확인.")
    else:
        st.caption("💡 `pip install streamlit-mermaid` 설치 시 Agent Graph 시각화 활성화")

# ── 탭: 단건 추론 / 비교 ─────────────────────────────────────
tab_infer, tab_cmp = st.tabs(["🔎 단건 추론", "⚖ forward vs agent 비교"])

with tab_infer:
    if run_btn:
        try:
            embedding = json.loads(embedding_input)
            if len(embedding) != 576:
                st.error(f"embedding 길이 {len(embedding)} ≠ 576")
                st.stop()
        except Exception:
            st.error("embedding JSON 파싱 실패.")
            st.stop()

        payload = {
            "purchase_value": purchase_value, "age": age, "sex": sex,
            "source": source, "browser": browser, "embedding": embedding,
            "experiment_case": EXPERIMENT_CASE,
            "percolate_version": percolate_version,
        }
        if weekday != "(없음)":
            payload["weekday_purchase"] = weekday
        if month != "(없음)":
            payload["month_purchase"] = month
        if ip_country.strip():
            payload["IP_country"] = ip_country.strip()

        with st.spinner("파이프라인 실행 중…"):
            try:
                result = call_pipeline(payload)
            except Exception as e:
                st.error(f"파이프라인 호출 실패: {e}")
                st.stop()
        st.session_state["last_result"] = result
        st.session_state["runs"][detect_mode(result)] = result   # 비교 탭 소스

    if st.session_state["last_result"]:
        render_result(st.session_state["last_result"])
    else:
        st.info("사이드바에서 거래 정보를 입력하고 **분석 실행**을 누르세요. "
                "결과는 서버 PIPELINE_MODE에 따라 forward/agent UI로 자동 렌더됩니다.")

with tab_cmp:
    render_compare()
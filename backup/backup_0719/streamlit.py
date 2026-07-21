"""
streamlit_app.py — Fraud Detection Pipeline UI (Agent RAG)
===========================================================
기존 forward RAG UI(orchestrator V5.0) 기반 유지.
Agent Loop 시각화 추가:
  1. Agent Trace 타임라인 (step-by-step 색상 구분)
  2. Latency Breakdown 바 차트 (plotly)
  3. LangGraph Mermaid 그래프 (사이드바 토글)
  4. 파이프라인 상세에 agent 필드 추가
     (planner_used / tool_execution_mode / agent_trace)
"""

import json
import os
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
# Helper — 기존 함수 유지
# ===========================

def call_pipeline(payload: dict) -> dict:
    resp = requests.post(
        f"{ORCHESTRATOR_URL}/detect",
        json=payload,
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def _badge(classification: str) -> str:
    return "🔴 ABNORMAL" if classification == "ABNORMAL" else "🟢 NORMAL"


def _format_latency(latency_ms: dict) -> str:
    """기존 forward RAG 형식 + agent 형식 양쪽 지원."""
    if not isinstance(latency_ms, dict):
        return str(latency_ms)
    # forward RAG 형식: router/retriever/analyzer/total
    forward_keys = ["router", "retriever", "analyzer", "total"]
    # agent 형식: base_retrieve/total
    agent_keys   = ["base_retrieve", "total"]

    parts = []
    for k in forward_keys:
        if k in latency_ms:
            parts.append(f"{k}: {latency_ms[k]:.0f}ms")
    if not parts:
        for k in agent_keys:
            if k in latency_ms:
                parts.append(f"{k}: {latency_ms[k]:.0f}ms")
    return "  |  ".join(parts)


# ===========================
# 신규 Helper — Agent Loop
# ===========================

# step별 아이콘 (agent_trace 항목과 매핑)
_TRACE_ICONS = {
    "pre_router:fast":             "⚡",
    "pre_router:deep":             "🔍",
    "pre_router":                  "🔍",
    "deep_retrieve":               "📥",
    "retrieve:base_reused":        "♻️",
    "retrieve":                    "📥",
    "analyze":                     "🧠",
    "plan:expand_within_cluster":  "🟡",
    "plan:reroute_cross_cluster":  "🟠",
    "plan:accept_verdict":         "🟢",
    "expand_within_cluster:direct":"🔄",
    "expand_within_cluster:mcp":   "🔄",
    "reroute_cross_cluster:direct":"↩️",
    "reroute_cross_cluster:mcp":   "↩️",
    "finalize":                    "✅",
}

_TRACE_COLORS = {
    "analyze":    "#4C78A8",
    "plan:expand":"#F58518",
    "plan:reroute":"#E45756",
    "plan:accept":"#54A24B",
    "finalize":   "#72B7B2",
}


def _render_agent_trace(trace: list) -> None:
    """Agent Trace를 step-by-step 타임라인으로 렌더링."""
    if not trace:
        return
    st.markdown("**🗺️ Agent Trace**")
    parts = []
    for step in trace:
        icon = next((v for k, v in _TRACE_ICONS.items() if k in step), "⚪")
        parts.append(f"{icon} `{step}`")
    st.markdown(" → ".join(parts))

    # planner 개입 여부 강조
    tool_steps = [s for s in trace if "expand_within_cluster" in s or "reroute_cross_cluster" in s]
    if tool_steps:
        st.info(f"🤖 Planner 개입: {' → '.join(tool_steps)}")


def _render_latency_chart(latency_ms: dict) -> None:
    """Latency Breakdown 바 차트 (plotly)."""
    if not PLOTLY_OK or not isinstance(latency_ms, dict):
        return

    # agent 형식: base_retrieve + tool_delta(= total - base_retrieve)
    total = latency_ms.get("total")
    base  = latency_ms.get("base_retrieve")

    # forward RAG 형식: router / retriever / analyzer
    forward_keys = ["router", "retriever", "analyzer"]
    forward_vals = [latency_ms.get(k) for k in forward_keys]
    has_forward  = any(v is not None for v in forward_vals)

    if has_forward:
        keys = [k for k, v in zip(forward_keys, forward_vals) if v is not None]
        vals = [v for v in forward_vals if v is not None]
        colors = ["#4C78A8", "#F58518", "#E45756"][:len(keys)]
    elif base is not None and total is not None:
        tool_delta = total - base
        keys   = ["base_retrieve", "tool_Δ\n(analyzer+planner)"]
        vals   = [base, tool_delta]
        colors = ["#9ECAE9", "#AEC7E8"]
    else:
        return

    fig = go.Figure(go.Bar(
        x=keys,
        y=vals,
        text=[f"{v:.0f}ms" for v in vals],
        textposition="auto",
        marker_color=colors,
        marker_line_color="white",
        marker_line_width=1.5,
    ))
    fig.update_layout(
        title="Latency Breakdown",
        height=240,
        margin=dict(t=36, b=16, l=16, r=16),
        yaxis_title="ms",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _fetch_mermaid() -> str | None:
    """Controller /graph/mermaid 엔드포인트 호출."""
    try:
        resp = requests.get(f"{CONTROLLER_URL}/graph/mermaid", timeout=5)
        resp.raise_for_status()
        return resp.json().get("mermaid")
    except Exception:
        return None


# ===========================
# Streamlit App
# ===========================

st.set_page_config(
    page_title="Fraud Detection — Agent RAG",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Fraud Detection Pipeline (Agent RAG)")
st.caption(
    f"Model: Qwen2.5-0.5B + LoRA ({ADAPTER_TYPE})  |  "
    f"Experiment: {EXPERIMENT_CASE}  |  "
    f"Orchestrator: {ORCHESTRATOR_URL}"
)

# ── Session 초기화 ──────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "ai",
            "content": (
                "안녕하세요! 거래 데이터를 입력하면 정상/이상 여부를 분석합니다.\n\n"
                "**입력 방법**: 사이드바에서 거래 필드를 직접 입력하거나, "
                "아래 채팅창에 JSON 형식으로 입력하세요.\n\n"
                "예시: `{\"purchase_value\": 1.5, \"age\": -0.5, \"sex\": \"F\", "
                "\"source\": \"SEO\", \"browser\": \"Chrome\", \"embedding\": [0.1, ...]}`"
            ),
        }
    ]

# ── 사이드바 ────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 거래 정보 입력")

    purchase_value = st.number_input(
        "purchase_value (z-score)", value=0.0, step=0.1, format="%.4f"
    )
    age = st.number_input(
        "age (z-score)", value=0.0, step=0.1, format="%.4f"
    )
    sex     = st.selectbox("sex",     ["M", "F"])
    source  = st.selectbox("source",  ["Direct", "Ads", "SEO"])
    browser = st.selectbox("browser", ["Chrome", "IE", "Safari", "FireFox", "Opera"])

    weekday = st.selectbox(
        "weekday_purchase (optional)",
        ["(없음)", "Monday", "Tuesday", "Wednesday", "Thursday",
         "Friday", "Saturday", "Sunday"],
    )
    month = st.selectbox(
        "month_purchase (optional)",
        ["(없음)", "January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"],
    )
    ip_country = st.text_input("IP_country (optional)", value="")

    st.divider()
    st.subheader("🔢 Embedding (576-dim)")
    embedding_input = st.text_area(
        "JSON array 형식으로 입력",
        height=100,
        placeholder="[0.123, -0.456, ...]",
    )

    percolate_version = st.selectbox(
        "Percolate Version",
        ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
         "v11", "v12", "v13", "v14"],
        index=11,   # v14 default
    )

    run_btn = st.button("🚀 분석 실행", use_container_width=True, type="primary")

    # ── LangGraph 시각화 토글 ──────────────────────────────
    st.divider()
    if MERMAID_OK:
        if st.toggle("🗺️ Agent Graph 보기", value=False):
            mermaid_txt = _fetch_mermaid()
            if mermaid_txt:
                st_mermaid(mermaid_txt, height=400)
            else:
                st.warning(
                    "Controller `/graph/mermaid` 응답 없음.\n"
                    "Controller pod가 실행 중인지 확인하세요."
                )
    else:
        st.caption("💡 `pip install streamlit-mermaid` 설치 시 Agent Graph 시각화 활성화")

# ── 채팅 히스토리 표시 ─────────────────────────────────────
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 사이드바 버튼 실행 ──────────────────────────────────────
if run_btn:
    try:
        embedding = json.loads(embedding_input)
        if len(embedding) != 576:
            st.sidebar.error(f"embedding 길이 {len(embedding)} ≠ 576")
            st.stop()
    except Exception:
        st.sidebar.error("embedding JSON 파싱 실패.")
        st.stop()

    payload = {
        "purchase_value":    purchase_value,
        "age":               age,
        "sex":               sex,
        "source":            source,
        "browser":           browser,
        "embedding":         embedding,
        "experiment_case":   EXPERIMENT_CASE,
        "percolate_version": percolate_version,
    }
    if weekday != "(없음)":
        payload["weekday_purchase"] = weekday
    if month != "(없음)":
        payload["month_purchase"] = month
    if ip_country.strip():
        payload["IP_country"] = ip_country.strip()

    input_summary = (
        f"**거래 분석 요청**\n"
        f"- purchase_value: `{purchase_value}`  age: `{age}`\n"
        f"- sex: `{sex}`  source: `{source}`  browser: `{browser}`\n"
        f"- percolate: `{percolate_version}`  experiment: `{EXPERIMENT_CASE}`"
    )
    with st.chat_message("human"):
        st.markdown(input_summary)
    st.session_state["messages"].append({"role": "human", "content": input_summary})

    with st.chat_message("ai"):
        with st.spinner("분석 중... (Local LLM 추론 포함, CPU 기준 30~120초 소요)"):
            try:
                result = call_pipeline(payload)

                cls   = result.get("classification", "UNKNOWN")
                conf  = result.get("confidence", 0)
                badge = _badge(cls)

                # ── 결과 카드 (col4: agent 정보 추가) ─────────
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                with col1:
                    st.metric("판정", badge)
                with col2:
                    st.metric("Confidence", f"{conf}%")
                with col3:
                    top1 = result.get("top_1_distance",
                                      result.get("top1_distance", -1))
                    st.metric("Top-1 Distance", f"{top1:.4f}")
                with col4:
                    planner_used = result.get("planner_used", False)
                    st.metric("Planner", "✅ 개입" if planner_used else "⬜ 미개입")

                # ── 추론 근거 ──────────────────────────────────
                if result.get("reasoning"):
                    with st.expander("📝 추론 근거", expanded=True):
                        st.write(result["reasoning"])
                        evidence = result.get("key_evidence", [])
                        if evidence:
                            st.markdown("**핵심 근거:**")
                            for ev in evidence:
                                st.markdown(f"- {ev}")
                        if result.get("final_verdict"):
                            st.markdown(f"**최종 판단:** {result['final_verdict']}")

                # ── Agent Trace 타임라인 (신규) ────────────────
                trace = result.get("agent_trace", [])
                if trace:
                    with st.expander("🔁 Agent Loop Trace", expanded=True):
                        _render_agent_trace(trace)

                # ── ★ [NB v3] 라운드별 판정 상세 — S4 1-key(det)에서 모델의
                #   유일한 자유 출력은 what_missing(감사 주석)이고 cls/suf는
                #   nd가 결정하므로, nd·sufficiency·wm을 라운드 단위로 노출.
                #   마지막 "final" 엔트리 = controller nd min-fusion 결과.
                _rounds = result.get("agent_rounds", [])
                if _rounds:
                    with st.expander("🧭 라운드별 판정 (nd / sufficiency / what_missing)"):
                        _rows = [{
                            "round":        r.get("round"),
                            "nd":           r.get("nd", r.get("nd_final")),
                            "cls":          r.get("cls"),
                            "sufficiency":  r.get("sufficiency"),
                            "what_missing": r.get("what_missing"),
                            "action":       r.get("action"),
                            "nd_round0":    r.get("nd_round0"),
                        } for r in _rounds]
                        st.table(_rows)

                # ── 파이프라인 상세 ────────────────────────────
                with st.expander("🔧 파이프라인 상세"):
                    meta_col1, meta_col2 = st.columns(2)
                    with meta_col1:
                        st.markdown(f"**Primary Cluster**: `{result.get('primary_cluster_id')}`")
                        st.markdown(f"**Vec Index**: `{result.get('vec_index')}`")
                        st.markdown(f"**Match Type**: `{result.get('match_type')}`")
                        # agent 신규 필드
                        st.markdown(f"**Track**: `{result.get('track', '-')}`")
                    with meta_col2:
                        st.markdown(f"**Planner Used**: `{result.get('planner_used', '-')}`")
                        st.markdown(f"**Tool Mode**: `{result.get('tool_execution_mode', '-')}`")
                        st.markdown(f"**Percolate**: `{result.get('percolate_version')}`")
                        st.markdown(f"**Experiment**: `{result.get('experiment_case')}`")

                    latency_str = _format_latency(result.get("latency_ms", {}))
                    st.markdown(f"**Latency**: `{latency_str}`")

                    # ── Latency Breakdown 차트 (신규) ──────────
                    _render_latency_chart(result.get("latency_ms", {}))

                    st.markdown("**Top-5 Clusters:**")
                    for c in result.get("top_5_clusters", []):
                        st.markdown(
                            f"  - cluster `{c['cluster_id']}` "
                            f"rank={c['rank']} score={c['score']:.3f}"
                        )

                # ── 채팅 메시지 저장 ───────────────────────────
                trace_str = " → ".join(trace) if trace else "-"
                response_text = (
                    f"**{badge}** (confidence: {conf}%)\n\n"
                    f"{result.get('reasoning', '')}\n\n"
                    f"**최종 판단**: {result.get('final_verdict', '')}\n\n"
                    f"**Agent Trace**: `{trace_str}`\n\n"
                    f"`latency: {latency_str}`"
                )
                st.session_state["messages"].append(
                    {"role": "ai", "content": response_text}
                )

            except requests.exceptions.ConnectionError:
                err = "❌ Orchestrator에 연결할 수 없습니다."
                st.error(err)
                st.session_state["messages"].append({"role": "ai", "content": err})
            except requests.exceptions.HTTPError as e:
                err = f"❌ Pipeline 오류: {e.response.status_code} — {e.response.text}"
                st.error(err)
                st.session_state["messages"].append({"role": "ai", "content": err})
            except Exception as e:
                err = f"❌ 예상치 못한 오류: {e}"
                st.error(err)
                st.session_state["messages"].append({"role": "ai", "content": err})

# ── 채팅창 JSON 직접 입력 ──────────────────────────────────
if chat_input := st.chat_input("JSON 형식으로 거래 데이터 직접 입력..."):
    with st.chat_message("human"):
        st.markdown(chat_input)
    st.session_state["messages"].append({"role": "human", "content": chat_input})

    with st.chat_message("ai"):
        try:
            payload = json.loads(chat_input)
            if "embedding" not in payload:
                st.warning("embedding 필드가 없습니다. 사이드바를 사용하세요.")
            else:
                payload.setdefault("experiment_case",   EXPERIMENT_CASE)
                payload.setdefault("percolate_version", "v14")

                with st.spinner("분석 중..."):
                    result = call_pipeline(payload)
                    cls    = result.get("classification", "UNKNOWN")
                    badge  = _badge(cls)
                    conf   = result.get("confidence", 0)
                    trace  = result.get("agent_trace", [])
                    latency_str = _format_latency(result.get("latency_ms", {}))

                    # 결과 카드
                    c1, c2, c3 = st.columns([2, 1, 1])
                    with c1: st.metric("판정", badge)
                    with c2: st.metric("Confidence", f"{conf}%")
                    with c3:
                        top1 = result.get("top_1_distance",
                                          result.get("top1_distance", -1))
                        st.metric("Top-1 Distance", f"{top1:.4f}")

                    if trace:
                        _render_agent_trace(trace)

                    _render_latency_chart(result.get("latency_ms", {}))

                    trace_str = " → ".join(trace) if trace else "-"
                    response_text = (
                        f"**{badge}** (confidence: {conf}%)\n\n"
                        f"{result.get('reasoning', '')}\n\n"
                        f"**Agent Trace**: `{trace_str}`\n\n"
                        f"`latency: {latency_str}`"
                    )
                    st.markdown(response_text)
                    st.session_state["messages"].append(
                        {"role": "ai", "content": response_text}
                    )

        except json.JSONDecodeError:
            msg = "❌ JSON 파싱 실패."
            st.error(msg)
            st.session_state["messages"].append({"role": "ai", "content": msg})
        except Exception as e:
            msg = f"❌ 오류: {e}"
            st.error(msg)
            st.session_state["messages"].append({"role": "ai", "content": msg})
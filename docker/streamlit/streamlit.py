"""
streamlit_app.py — Fraud Detection Pipeline UI
================================================
chat.py 구조를 기반으로, orchestrator V5.0 파이프라인과 연동.
docs embedder / eventrac 기능 제외.
"""

import json
import os
import requests
import streamlit as st
import yaml

# ===========================
# Config
# ===========================

_CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
with open(_CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", CFG.get("orchestrator_url", "http://anomaly-orchestrator:8080"))
EXPERIMENT_CASE  = CFG["experiment_case"]
ADAPTER_TYPE     = CFG["adapter_type"]

# ===========================
# Helper
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
    if classification == "ABNORMAL":
        return "🔴 ABNORMAL"
    return "🟢 NORMAL"


def _format_latency(latency_ms: dict) -> str:
    parts = []
    for k in ["router", "retriever", "analyzer", "total"]:
        if k in latency_ms:
            parts.append(f"{k}: {latency_ms[k]:.0f}ms")
    return "  |  ".join(parts)


# ===========================
# Streamlit App
# ===========================

st.set_page_config(
    page_title="Fraud Detection — Local LLM Analyzer",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Fraud Detection Pipeline")
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

# ── 사이드바: 거래 필드 직접 입력 ───────────────────────────
with st.sidebar:
    st.header("📋 거래 정보 입력")

    purchase_value = st.number_input(
        "purchase_value (z-score)", value=0.0, step=0.1, format="%.4f"
    )
    age = st.number_input(
        "age (z-score)", value=0.0, step=0.1, format="%.4f"
    )
    sex = st.selectbox("sex", ["M", "F"])
    source = st.selectbox("source", ["Direct", "Ads", "SEO"])
    browser = st.selectbox("browser", ["Chrome", "IE", "Safari", "FireFox", "Opera"])

    weekday = st.selectbox(
        "weekday_purchase (optional)",
        ["(없음)", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
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
        placeholder='[0.123, -0.456, ...]',
    )

    percolate_version = st.selectbox(
        "Percolate Version",
        ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"],
        index=3,  # v4 default
    )

    run_btn = st.button("🚀 분석 실행", use_container_width=True, type="primary")

# ── 채팅 히스토리 표시 ─────────────────────────────────────
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 사이드바 버튼 실행 ──────────────────────────────────────
if run_btn:
    # embedding 파싱
    try:
        embedding = json.loads(embedding_input)
        if len(embedding) != 576:
            st.sidebar.error(f"embedding 길이가 {len(embedding)}입니다. 576이어야 합니다.")
            st.stop()
    except Exception:
        st.sidebar.error("embedding JSON 파싱 실패. 올바른 배열 형식인지 확인하세요.")
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

    # 입력 메시지 표시
    input_summary = (
        f"**거래 분석 요청**\n"
        f"- purchase_value: `{purchase_value}`  age: `{age}`\n"
        f"- sex: `{sex}`  source: `{source}`  browser: `{browser}`\n"
        f"- percolate: `{percolate_version}`  experiment: `{EXPERIMENT_CASE}`"
    )
    with st.chat_message("human"):
        st.markdown(input_summary)
    st.session_state["messages"].append({"role": "human", "content": input_summary})

    # 파이프라인 호출
    with st.chat_message("ai"):
        with st.spinner("분석 중... (Local LLM 추론 포함, CPU 기준 30~60초 소요)"):
            try:
                result = call_pipeline(payload)

                cls   = result["classification"]
                conf  = result["confidence"]
                badge = _badge(cls)

                # ── 결과 카드 ──────────────────────────────────
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.metric("판정", badge)
                with col2:
                    st.metric("Confidence", f"{conf}%")
                with col3:
                    st.metric("Top-1 Distance", f"{result.get('top1_distance', -1):.4f}")

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

                # ── 파이프라인 메타 ────────────────────────────
                with st.expander("🔧 파이프라인 상세"):
                    meta_col1, meta_col2 = st.columns(2)
                    with meta_col1:
                        st.markdown(f"**Primary Cluster**: `{result.get('primary_cluster_id')}`")
                        st.markdown(f"**Vec Index**: `{result.get('vec_index')}`")
                        st.markdown(f"**Match Type**: `{result.get('match_type')}`")
                        st.markdown(f"**Skip Analyzer**: `{result.get('skip_analyzer')}`")
                    with meta_col2:
                        st.markdown(f"**Persona Used**: `{result.get('persona_used')}`")
                        st.markdown(f"**Page Fault Δ**: `{result.get('page_fault_delta', 0)}`")
                        st.markdown(f"**Percolate**: `{result.get('percolate_version')}`")
                        st.markdown(f"**Experiment**: `{result.get('experiment_case')}`")

                    latency_str = _format_latency(result.get("latency_ms", {}))
                    st.markdown(f"**Latency**: `{latency_str}`")

                    st.markdown("**Top-5 Clusters:**")
                    for c in result.get("top_5_clusters", []):
                        st.markdown(
                            f"  - cluster `{c['cluster_id']}` "
                            f"rank={c['rank']} score={c['score']:.3f}"
                        )

                # ── 채팅 메시지로 저장 ─────────────────────────
                response_text = (
                    f"**{badge}** (confidence: {conf}%)\n\n"
                    f"{result.get('reasoning', '')}\n\n"
                    f"**최종 판단**: {result.get('final_verdict', '')}\n\n"
                    f"`latency: {_format_latency(result.get('latency_ms', {}))}`"
                )
                st.session_state["messages"].append(
                    {"role": "ai", "content": response_text}
                )

            except requests.exceptions.ConnectionError:
                err = "❌ Orchestrator에 연결할 수 없습니다. 서비스 상태를 확인하세요."
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
                payload.setdefault("percolate_version", "v4")

                with st.spinner("분석 중..."):
                    result = call_pipeline(payload)
                    cls    = result["classification"]
                    badge  = _badge(cls)
                    conf   = result["confidence"]

                    response_text = (
                        f"**{badge}** (confidence: {conf}%)\n\n"
                        f"{result.get('reasoning', '')}\n\n"
                        f"**최종 판단**: {result.get('final_verdict', '')}\n\n"
                        f"`latency: {_format_latency(result.get('latency_ms', {}))}`"
                    )
                    st.markdown(response_text)
                    st.session_state["messages"].append(
                        {"role": "ai", "content": response_text}
                    )

        except json.JSONDecodeError:
            msg = "❌ JSON 파싱 실패. 올바른 형식인지 확인하세요."
            st.error(msg)
            st.session_state["messages"].append({"role": "ai", "content": msg})

        except Exception as e:
            msg = f"❌ 오류: {e}"
            st.error(msg)
            st.session_state["messages"].append({"role": "ai", "content": msg})
"""
Analyzer Service - v3.0
========================

v3.0 변경사항:
  ★ query_id 완전 제거 (AnalysisRequest / AnalysisResponse)
      - router v5에서 query_id 제거됨 → 파이프라인 전체 일관성
      - 내부 로깅용 request_id는 서비스 내부에서 생성
  ★ leaf_id / support 제거 (AnalysisRequest)
      - router v5 ClusterCandidate에서 제거됨
      - 프롬프트에도 미사용 → 완전 제거
  ★ top_k_normal_results: 필드명 변경 (top_k_normal_transactions → top_k_normal_results)
      - retriever v5 SearchResult 포맷에 맞춤:
        {es_doc_id, original_index, cluster_id, distance, score}
      - format_retrieved_results() 신규 (기존 format_normal_transactions 대체)
  ★ top1_distance 추가 (AnalysisRequest)
      - retriever top1_distance → 프롬프트에 컨텍스트로 포함
      - distance 기반 이상 신호를 LLM에 전달

v2.2 유지:
  - PERSONA_ENABLED: default True
  - TREE_RULES_ENABLED: default False
  - JSON mode / 동적 프롬프트 구성 (4가지 조합)
  - Rate limiter / LLM retry / Prometheus 메트릭
"""

import os
import time
import json
import uuid
from typing import Dict, List, Optional
from collections import deque
from threading import Lock

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import structlog
import asyncio

from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import ChatPromptTemplate

# ===========================
# Configuration
# ===========================

LLM_MODEL            = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
LLM_TEMPERATURE      = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS       = int(os.getenv("LLM_MAX_TOKENS", "1024"))

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_REGION  = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")

MAX_RETRIES          = int(os.getenv("MAX_RETRIES", "5"))
RETRY_BACKOFF        = int(os.getenv("RETRY_BACKOFF", "5"))
REQUEST_TIMEOUT      = int(os.getenv("REQUEST_TIMEOUT", "120"))

MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "30"))

PERSONA_ENABLED    = os.getenv("PERSONA_ENABLED",    "true").lower()  == "true"
TREE_RULES_ENABLED = os.getenv("TREE_RULES_ENABLED", "false").lower() == "true"
JSON_MODE_ENABLED  = os.getenv("JSON_MODE_ENABLED",  "true").lower()  == "true"

# ===========================
# Rate Limiter
# ===========================

class RateLimiter:
    def __init__(self, max_calls: int, time_window: int = 60):
        self.max_calls   = max_calls
        self.time_window = time_window
        self.calls       = deque()
        self.lock        = Lock()

    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            while self.calls and self.calls[0] < now - self.time_window:
                self.calls.popleft()
            if len(self.calls) >= self.max_calls:
                wait = self.time_window - (now - self.calls[0])
                if wait > 0:
                    logger.warning("rate_limit_wait", wait_seconds=wait)
                    time.sleep(wait)
                    now = time.time()
                    while self.calls and self.calls[0] < now - self.time_window:
                        self.calls.popleft()
            self.calls.append(time.time())

rate_limiter = RateLimiter(max_calls=MAX_REQUESTS_PER_MINUTE)

# ===========================
# FastAPI & Logger
# ===========================

app    = FastAPI(title="Anomaly Analyzer v3.0", version="3.0.0")
logger = structlog.get_logger()

# ===========================
# Prometheus Metrics
# ===========================

analysis_requests_total = Counter(
    "analysis_requests_total", "Total analysis requests", ["status", "experiment_case"]
)
llm_inference_latency = Histogram(
    "llm_inference_latency_seconds", "LLM inference latency",
    ["experiment_case"],
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)
llm_requests_per_second = Counter("llm_requests_per_second", "LLM requests for HPA")
classification_results  = Counter(
    "classification_results_total", "Classification results", ["result", "experiment_case"]
)
rate_limit_waits  = Counter("rate_limit_waits_total", "Rate limiter waits")
llm_retry_counter = Counter("llm_retry_attempts_total", "LLM retry attempts", ["attempt"])
persona_used_cnt  = Counter("persona_used_total", "Persona delivered", ["experiment_case"])
tree_rules_cnt    = Counter("tree_rules_used_total", "Tree rules included", ["experiment_case"])
json_mode_counter = Counter("json_mode_responses_total", "JSON mode responses", ["status"])

# ===========================
# Pydantic Models
# ===========================

class AnalysisRequest(BaseModel):
    """
    ★ v3.0:
      - query_id 제거 (router v5 일관성)
      - leaf_id / support 제거 (router v5 ClusterCandidate에서 제거됨)
      - top_k_normal_results: retriever v5 SearchResult 리스트
          {es_doc_id, original_index, cluster_id, distance, score}
      - top1_distance: retriever top-1 cosine distance (LLM 컨텍스트용)
    """
    test_transaction: Dict                 = Field(..., description="원본 거래 데이터")
    cluster_id:       int                  = Field(..., description="매칭된 클러스터 ID")
    top_k_normal_results: List[Dict]       = Field(
        ..., description="Retriever top-K 결과: {es_doc_id, original_index, cluster_id, distance, score}"
    )
    top1_distance:    float                = Field(0.0, description="Retriever top-1 cosine distance")
    persona:          Optional[Dict]       = Field(None, description="클러스터 persona")
    tree_features:    Optional[Dict]       = Field(None, description="percolate rule range 조건 (임베딩 공간)")


class AnalysisResponse(BaseModel):
    """
    ★ v3.0: query_id 제거
    """
    classification:  str            # NORMAL | ABNORMAL
    confidence:      int            # 0-100
    reasoning:       str
    key_evidence:    List[str]
    final_verdict:   str
    latency_ms:      float
    persona_used:    bool  = False
    tree_rules_used: bool  = False
    experiment_case: str

# ===========================
# Vertex AI Client
# ===========================

try:
    if JSON_MODE_ENABLED:
        vertexAI = ChatVertexAI(
            model_name=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
            streaming=False,
            convert_system_message_to_human=True,
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_REGION,
            model_kwargs={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "classification": {"type": "string", "enum": ["NORMAL", "ABNORMAL"]},
                        "confidence":     {"type": "integer", "minimum": 0, "maximum": 100},
                        "reasoning":      {"type": "string"},
                        "key_evidence":   {"type": "array", "items": {"type": "string"}, "maxItems": 3},
                        "final_verdict":  {"type": "string"},
                    },
                    "required": ["classification", "confidence", "reasoning",
                                 "key_evidence", "final_verdict"],
                }
            }
        )
    else:
        vertexAI = ChatVertexAI(
            model_name=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
            streaming=False,
            convert_system_message_to_human=True,
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_REGION,
        )
    logger.info(
        "vertex_ai_initialized",
        model=LLM_MODEL,
        json_mode=JSON_MODE_ENABLED,
        persona_enabled=PERSONA_ENABLED,
        tree_rules_enabled=TREE_RULES_ENABLED,
    )
except Exception as e:
    logger.error("vertex_ai_initialization_failed", error=str(e))
    raise

# ===========================
# Helpers: Data Formatting
# ===========================

def format_transaction_data(data: Dict) -> str:
    """거래 데이터를 LLM이 읽기 좋은 섹션 형태로 포맷팅."""
    if not data:
        return "No data provided."
    sections = {
        "Numeric Features":     ["purchase_value", "age"],
        "Categorical Features": ["sex", "source", "browser"],
        "Additional Context":   ["weekday_purchase", "month_purchase", "IP_country"],
    }
    lines = []
    for section, keys in sections.items():
        existing = [k for k in keys if k in data and data[k] is not None]
        if existing:
            lines.append(f"[{section}]")
            for k in existing:
                v = data[k]
                lines.append(f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}")
            lines.append("")
    return "\n".join(lines).strip()


def format_persona(persona: Optional[Dict]) -> str:
    """
    Cluster persona 포맷팅.
    구조: {cluster_id, size, numeric_stats, categorical_distribution, description}
    """
    if not persona:
        return "N/A"
    lines = []
    if "description" in persona:
        lines.append(f"**Summary**: {persona['description']}")
        lines.append("")
    if persona.get("numeric_stats"):
        lines.append("**Numeric Statistics:**")
        for feat, stats in persona["numeric_stats"].items():
            mean = stats.get("mean", 0)
            std  = stats.get("std",  0)
            lines.append(f"  - {feat}: mean={mean:.3f}, std={std:.3f}, "
                         f"min={stats.get('min', 0):.3f}, max={stats.get('max', 0):.3f}")
        lines.append("")
    if persona.get("categorical_distribution"):
        lines.append("**Categorical Distribution:**")
        for feat, info in persona["categorical_distribution"].items():
            dominant = info.get("dominant", "?")
            freq     = info.get("frequency", 0)
            dist     = info.get("distribution", {})
            dist_str = ", ".join(f"{k}:{v}" for k, v in list(dist.items())[:4])
            lines.append(f"  - {feat}: dominant={dominant} ({freq:.0%})  [{dist_str}]")
        lines.append("")
    return "\n".join(lines).strip()


def format_tree_features(features: Optional[Dict]) -> str:
    """
    Tree percolate rule range 조건 포맷팅.
    ⚠️ v1~v576 임베딩 공간 기반 조건 (원본 피처 값과 무관).
    """
    if not features:
        return "N/A"
    lines = ["⚠️ Embedding-space range conditions (v1~vN dimensions), not raw feature values."]
    for key, value in features.items():
        lines.append(f"  - {key}: {value}" if not isinstance(value, dict) else f"  - {key}: {value}")
    return "\n".join(lines)


def format_retrieved_results(results: List[Dict], top1_distance: float) -> str:
    """
    ★ v3.0: retriever v5 SearchResult 포맷 처리.
    SearchResult: {es_doc_id, original_index, cluster_id, distance, score}

    top1_distance를 함께 표시하여 LLM이 거리 기반 이상 신호를 이해하도록 함.
    """
    if not results:
        return "No retrieved results."

    lines = [
        f"Top-1 cosine distance: {top1_distance:.4f}  "
        f"({'close match ✓' if top1_distance < 0.3 else 'moderate match' if top1_distance < 0.6 else 'distant match ⚠'})",
        "",
    ]
    for i, r in enumerate(results[:5], 1):
        dist  = r.get("distance", 0.0)
        score = r.get("score",    0.0)
        match_signal = "✓" if dist < 0.3 else ("~" if dist < 0.6 else "⚠")
        lines.append(
            f"Result {i}: original_index={r.get('original_index', '?')}  "
            f"cluster_id={r.get('cluster_id', '?')}  "
            f"distance={dist:.4f}  score={score:.4f}  {match_signal}"
        )
    return "\n".join(lines)

# ===========================
# Dynamic Prompt Builder
# ===========================

SYSTEM_MSG = (
    "You are an expert fraud detection analyst. "
    "Analyze the TEST TRANSACTION by comparing it with the cluster profile and retrieved similar patterns."
)


def build_human_prompt(
    test_data_fmt:     str,
    cluster_id:        int,
    retrieved_fmt:     str,
    persona_fmt:       Optional[str],
    tree_features_fmt: Optional[str],
    use_persona:       bool,
    use_tree_rules:    bool,
) -> str:
    """
    ★ v3.0: query_id 파라미터 제거.
    persona / tree_rules 포함 여부에 따라 동적으로 프롬프트 구성.

    분석 프레임워크:
      - Persona O + Tree Rules O: 3-step (Statistics / Boundary / Pattern)
      - Persona O + Tree Rules X: 2-step (Statistics / Pattern)  ← Default
      - Persona X + Tree Rules O: 2-step (Boundary / Pattern)
      - Persona X + Tree Rules X: 1-step (Pattern)
    """
    sections = []

    # ── TEST TRANSACTION ────────────────────────────────────────
    sections.append(f"""## TEST TRANSACTION (TO ANALYZE)
{test_data_fmt}""")

    # ── CLUSTER PROFILE ─────────────────────────────────────────
    cluster_block = [f"## CLUSTER #{cluster_id} PROFILE"]
    if use_persona and persona_fmt:
        cluster_block.append("\n**Cluster Statistical Profile:**")
        cluster_block.append(persona_fmt)
    if use_tree_rules and tree_features_fmt:
        cluster_block.append("\n**Decision Tree Rules (Embedding Space Range Conditions):**")
        cluster_block.append(tree_features_fmt)
    if not use_persona and not use_tree_rules:
        cluster_block.append(f"\nCluster ID: {cluster_id} (no additional profile available)")
    sections.append("\n".join(cluster_block))

    # ── RETRIEVED SIMILAR PATTERNS ──────────────────────────────
    sections.append(f"""## RETRIEVED SIMILAR PATTERNS (FROM VECTOR SEARCH)
{retrieved_fmt}""")

    # ── ANALYSIS TASK ────────────────────────────────────────────
    steps = []
    step_num = 1
    if use_persona:
        steps.append(
            f"{step_num}. **Statistical Deviation**: "
            "Compare test values against the cluster's numeric/categorical profile."
        )
        step_num += 1
    if use_tree_rules:
        steps.append(
            f"{step_num}. **Boundary Check**: "
            "Does the test satisfy the embedding-space decision tree rules?"
        )
        step_num += 1
    steps.append(
        f"{step_num}. **Pattern Similarity**: "
        "Interpret the cosine distance scores — low distance = similar to known normal patterns, "
        "high distance = potentially anomalous."
    )
    framework_str = "\n".join(steps)

    sections.append(f"""## ANALYSIS TASK
Determine if the test transaction is **NORMAL** or **ABNORMAL**.

**Analysis Framework:**
{framework_str}

**Output Format (JSON):**
```json
{{{{
  "classification": "NORMAL" or "ABNORMAL",
  "confidence": 0-100,
  "reasoning": "Brief analysis following the framework above",
  "key_evidence": ["Point 1", "Point 2", "Point 3"],
  "final_verdict": "One concise sentence summary"
}}}}
```

**Guidelines:**
- ABNORMAL if test deviates significantly from cluster profile or retrieved patterns
- NORMAL if test aligns with known cluster patterns
- High top-1 distance (> 0.6) is a strong anomaly signal
- Confidence: 0=uncertain, 100=certain
- Reference specific values in reasoning""")

    return "\n\n---\n\n".join(sections)

# ===========================
# Response Parsing
# ===========================

def parse_json_response(response_text: str) -> Dict:
    try:
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        parsed = json.loads(text)
        required = ["classification", "confidence", "reasoning", "key_evidence", "final_verdict"]
        for k in required:
            if k not in parsed:
                raise ValueError(f"Missing key: {k}")

        parsed["classification"] = parsed["classification"].upper()
        if parsed["classification"] not in ("NORMAL", "ABNORMAL"):
            parsed["classification"] = "ABNORMAL"
        if not isinstance(parsed["key_evidence"], list):
            parsed["key_evidence"] = [str(parsed["key_evidence"])]
        parsed["key_evidence"] = parsed["key_evidence"][:3]

        json_mode_counter.labels(status="success").inc()
        return parsed

    except Exception as e:
        json_mode_counter.labels(status="parse_error").inc()
        logger.error("json_parse_error", error=str(e), response=response_text[:200])
        return {
            "classification": "ABNORMAL",
            "confidence":     50,
            "reasoning":      f"JSON parsing failed: {e}",
            "key_evidence":   ["Parse error occurred"],
            "final_verdict":  "Unable to determine classification",
        }

# ===========================
# Core Analysis Logic
# ===========================

def _run_llm(prompt_input: Dict, experiment_case: str) -> Dict:
    """LLM 호출 + retry + rate limiting."""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MSG),
        ("human",  "{human_msg}"),
    ])
    chain = prompt_template | vertexAI

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            rate_limiter.wait_if_needed()
            with llm_inference_latency.labels(experiment_case=experiment_case).time():
                response = chain.invoke(prompt_input)
                llm_requests_per_second.inc()
            return response
        except Exception as e:
            last_error = e
            llm_retry_counter.labels(attempt=str(attempt + 1)).inc()
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF * (2 ** attempt)
                logger.warning("llm_retry", attempt=attempt + 1, wait=wait, error=str(e))
                time.sleep(wait)

    raise last_error


def _do_analyze(
    request:         AnalysisRequest,
    experiment_case: str,
    use_persona:     bool,
    use_tree_rules:  bool,
) -> AnalysisResponse:
    """단일 분석 처리 (analyze / analyze-batch 공통)."""
    start = time.time()

    # 내부 추적용 request_id (query_id 대체)
    request_id = str(uuid.uuid4())[:8]

    # ── 데이터 포맷팅 ────────────────────────────────────────────
    test_data_fmt = format_transaction_data(request.test_transaction)

    # ★ v3.0: retriever v5 결과 포맷
    retrieved_fmt = format_retrieved_results(request.top_k_normal_results, request.top1_distance)

    persona_fmt = None
    if use_persona and request.persona:
        persona_fmt = format_persona(request.persona)
        persona_used_cnt.labels(experiment_case=experiment_case).inc()

    tree_features_fmt = None
    if use_tree_rules and request.tree_features:
        tree_features_fmt = format_tree_features(request.tree_features)
        tree_rules_cnt.labels(experiment_case=experiment_case).inc()

    # ── 동적 프롬프트 빌드 ───────────────────────────────────────
    human_msg = build_human_prompt(
        test_data_fmt=test_data_fmt,
        cluster_id=request.cluster_id,
        retrieved_fmt=retrieved_fmt,
        persona_fmt=persona_fmt,
        tree_features_fmt=tree_features_fmt,
        use_persona=use_persona and persona_fmt is not None,
        use_tree_rules=use_tree_rules and tree_features_fmt is not None,
    )

    logger.info(
        "analysis_start",
        request_id=request_id,
        cluster_id=request.cluster_id,
        top1_distance=request.top1_distance,
        n_results=len(request.top_k_normal_results),
        persona_used=use_persona and persona_fmt is not None,
        tree_rules_used=use_tree_rules and tree_features_fmt is not None,
        experiment_case=experiment_case,
    )

    # ── LLM 호출 ────────────────────────────────────────────────
    response = _run_llm({"human_msg": human_msg}, experiment_case)

    # ── 파싱 ────────────────────────────────────────────────────
    if JSON_MODE_ENABLED:
        parsed = parse_json_response(response.content)
    else:
        parsed = {
            "classification": "ABNORMAL",
            "confidence":     50,
            "reasoning":      response.content,
            "key_evidence":   ["Text mode — no structured parsing"],
            "final_verdict":  "See reasoning",
        }

    total_ms = (time.time() - start) * 1000

    analysis_requests_total.labels(status="success", experiment_case=experiment_case).inc()
    classification_results.labels(result=parsed["classification"],
                                   experiment_case=experiment_case).inc()

    logger.info(
        "analysis_done",
        request_id=request_id,
        classification=parsed["classification"],
        confidence=parsed["confidence"],
        top1_distance=request.top1_distance,
        latency_ms=total_ms,
        experiment_case=experiment_case,
    )

    return AnalysisResponse(
        classification=parsed["classification"],
        confidence=parsed["confidence"],
        reasoning=parsed["reasoning"],
        key_evidence=parsed["key_evidence"],
        final_verdict=parsed["final_verdict"],
        latency_ms=total_ms,
        persona_used=use_persona and persona_fmt is not None,
        tree_rules_used=use_tree_rules and tree_features_fmt is not None,
        experiment_case=experiment_case,
    )

# ===========================
# API Endpoints
# ===========================

# @app.post("/analyze", response_model=AnalysisResponse)
# async def analyze_transaction(
#     request: AnalysisRequest,
#     x_experiment_case:    Optional[str] = Header(None, description="Override experiment_case"),
#     x_persona_enabled:    Optional[str] = Header(None, description="Override PERSONA_ENABLED (true/false)"),
#     x_tree_rules_enabled: Optional[str] = Header(None, description="Override TREE_RULES_ENABLED (true/false)"),
# ):
#     """
#     LLM 기반 이상 탐지 분석 (단건).
#     플래그 우선순위: Request Header > ConfigMap 환경변수
#     """
#     exp_case       = x_experiment_case or os.getenv("EXPERIMENT_CASE", "pca_64")
#     use_persona    = (x_persona_enabled.lower()    == "true") if x_persona_enabled    else PERSONA_ENABLED
#     use_tree_rules = (x_tree_rules_enabled.lower() == "true") if x_tree_rules_enabled else TREE_RULES_ENABLED

#     try:
#         return _do_analyze(request, exp_case, use_persona, use_tree_rules)
#     except Exception as e:
#         analysis_requests_total.labels(status="error", experiment_case=exp_case).inc()
#         logger.error("analyze_failed", error=str(e))
#         raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_transaction(
    request: AnalysisRequest,
    x_experiment_case:    Optional[str] = Header(None),
    x_persona_enabled:    Optional[str] = Header(None),
    x_tree_rules_enabled: Optional[str] = Header(None),
):
    exp_case       = x_experiment_case or os.getenv("EXPERIMENT_CASE", "pca_64")
    use_persona    = (x_persona_enabled.lower()    == "true") if x_persona_enabled    else PERSONA_ENABLED
    use_tree_rules = (x_tree_rules_enabled.lower() == "true") if x_tree_rules_enabled else TREE_RULES_ENABLED

    try:
        return await asyncio.to_thread(          # ← 추가
            _do_analyze, request, exp_case, use_persona, use_tree_rules
        )
    except Exception as e:
        analysis_requests_total.labels(status="error", experiment_case=exp_case).inc()
        logger.error("analyze_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.post("/analyze-batch", response_model=List[AnalysisResponse])
async def analyze_batch(
    requests: List[AnalysisRequest],
    x_experiment_case:    Optional[str] = Header(None),
    x_persona_enabled:    Optional[str] = Header(None),
    x_tree_rules_enabled: Optional[str] = Header(None),
):
    """LLM 기반 이상 탐지 배치 분석 (최대 20건). LangChain abatch 병렬 처리."""
    if len(requests) > 20:
        raise HTTPException(status_code=400, detail="Batch size exceeds limit (max 20)")

    exp_case       = x_experiment_case or os.getenv("EXPERIMENT_CASE", "pca_64")
    use_persona    = (x_persona_enabled.lower()    == "true") if x_persona_enabled    else PERSONA_ENABLED
    use_tree_rules = (x_tree_rules_enabled.lower() == "true") if x_tree_rules_enabled else TREE_RULES_ENABLED

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MSG),
        ("human",  "{human_msg}"),
    ])
    chain = prompt_template | vertexAI

    batch_start   = time.time()
    prompt_inputs: List[Dict] = []

    for req in requests:
        test_data_fmt  = format_transaction_data(req.test_transaction)
        retrieved_fmt  = format_retrieved_results(req.top_k_normal_results, req.top1_distance)
        persona_fmt    = format_persona(req.persona) if (use_persona and req.persona) else None
        tree_fmt       = format_tree_features(req.tree_features) if (use_tree_rules and req.tree_features) else None

        human_msg = build_human_prompt(
            test_data_fmt=test_data_fmt,
            cluster_id=req.cluster_id,
            retrieved_fmt=retrieved_fmt,
            persona_fmt=persona_fmt,
            tree_features_fmt=tree_fmt,
            use_persona=use_persona and persona_fmt is not None,
            use_tree_rules=use_tree_rules and tree_fmt is not None,
        )
        prompt_inputs.append({"human_msg": human_msg})

    rate_limiter.wait_if_needed()
    llm_start  = time.time()
    responses  = await chain.abatch(prompt_inputs)
    llm_ms     = (time.time() - llm_start) * 1000

    results: List[AnalysisResponse] = []
    for i, response in enumerate(responses):
        req = requests[i]
        try:
            parsed = parse_json_response(response.content) if JSON_MODE_ENABLED else {
                "classification": "ABNORMAL", "confidence": 50,
                "reasoning": response.content,
                "key_evidence": ["Text mode"], "final_verdict": "See reasoning",
            }
            results.append(AnalysisResponse(
                classification=parsed["classification"],
                confidence=parsed["confidence"],
                reasoning=parsed["reasoning"],
                key_evidence=parsed["key_evidence"],
                final_verdict=parsed["final_verdict"],
                latency_ms=llm_ms / len(requests),
                persona_used=use_persona and req.persona is not None,
                tree_rules_used=use_tree_rules and req.tree_features is not None,
                experiment_case=exp_case,
            ))
            analysis_requests_total.labels(status="success", experiment_case=exp_case).inc()
        except Exception as e:
            logger.error("batch_item_error", index=i, error=str(e))
            results.append(AnalysisResponse(
                classification="ABNORMAL", confidence=50,
                reasoning=f"Error: {e}", key_evidence=["Error"],
                final_verdict="Unable to classify",
                latency_ms=0.0, persona_used=False, tree_rules_used=False,
                experiment_case=exp_case,
            ))

    logger.info("batch_done", batch_size=len(requests),
                total_ms=(time.time() - batch_start) * 1000, llm_ms=llm_ms)
    return results

# ===========================
# Health / Config / Metrics
# ===========================

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "analyzer", "version": "3.0.0"}


@app.get("/ready")
async def ready():
    try:
        if vertexAI is None:
            raise HTTPException(status_code=503, detail="Vertex AI not initialized")
        return {
            "status": "ready",
            "vertex_ai": {
                "model":     LLM_MODEL,
                "project":   GOOGLE_CLOUD_PROJECT,
                "region":    GOOGLE_CLOUD_REGION,
                "json_mode": JSON_MODE_ENABLED,
            },
            "configuration": {
                "persona_enabled":    PERSONA_ENABLED,
                "tree_rules_enabled": TREE_RULES_ENABLED,
                "max_rpm":            MAX_REQUESTS_PER_MINUTE,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/config")
async def get_config():
    return {
        "llm":   {"model": LLM_MODEL, "temperature": LLM_TEMPERATURE, "max_tokens": LLM_MAX_TOKENS},
        "flags": {
            "persona_enabled":    PERSONA_ENABLED,
            "tree_rules_enabled": TREE_RULES_ENABLED,
            "json_mode":          JSON_MODE_ENABLED,
        },
        "note": {
            "persona":    "Cluster statistical profile — meaningful for LLM context",
            "tree_rules": "Embedding-space range conditions (v1~vN) — debugging only",
            "top_k_normal_results": "Retriever v5 SearchResult: es_doc_id, original_index, cluster_id, distance, score",
        },
        "rate_limiting": {"max_rpm": MAX_REQUESTS_PER_MINUTE},
        "retry":         {"max_retries": MAX_RETRIES, "backoff": RETRY_BACKOFF},
    }


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.on_event("startup")
async def startup_event():
    logger.info(
        "analyzer_started",
        version="3.0.0",
        model=LLM_MODEL,
        persona_enabled=PERSONA_ENABLED,
        tree_rules_enabled=TREE_RULES_ENABLED,
        json_mode=JSON_MODE_ENABLED,
        project=GOOGLE_CLOUD_PROJECT,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
